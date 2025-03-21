//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_OPTIMIZECONCAT
#define GEN_PASS_DEF_OPTIMIZECONCAT
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

//
// EliminateConcat
//

class EliminateConcat final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    EliminateConcat(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        setDebugName("EliminateConcat");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EliminateConcat::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }

    const auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsets().value());
    DenseMap<VPU::SliceOp, std::pair<SmallVector<int64_t>, mlir::Value>> newSliceOffsetsInputMap;

    const auto allUsersSliceSubTensors = llvm::all_of(origOp->getUsers(), [&](auto userOp) {
        auto maybeSliceOp = mlir::dyn_cast_or_null<VPU::SliceOp>(userOp);
        if (maybeSliceOp == nullptr) {
            return false;
        }

        auto sliceOffset = parseIntArrayAttr<int64_t>(maybeSliceOp.getStaticOffsets());
        const auto sliceOutShape = getShape(maybeSliceOp.getResult()).raw();

        for (const auto& p : zip(origOp.getInputs(), concatOffsets)) {
            const auto concatInput = std::get<0>(p);
            const auto concatInputShape = getShape(concatInput).raw();
            const auto concatOffset = std::get<1>(p);

            if (auto inputOp = concatInput.getDefiningOp()) {
                if (!inputOp->hasOneUse()) {
                    continue;
                }
            }

            const auto isSubTensor = [&]() -> bool {
                for (const auto dim : irange(sliceOutShape.size())) {
                    if ((sliceOffset[dim] < concatOffset[dim]) ||
                        (concatOffset[dim] + concatInputShape[dim] < sliceOffset[dim] + sliceOutShape[dim])) {
                        return false;
                    }
                }
                return true;
            };

            if (!isSubTensor()) {
                continue;
            }

            for (const auto dim : irange(sliceOffset.size())) {
                sliceOffset[dim] -= concatOffset[dim];
            }

            newSliceOffsetsInputMap[maybeSliceOp] = std::pair{sliceOffset, concatInput};
            return true;
        }

        return false;
    });

    if (!allUsersSliceSubTensors) {
        return mlir::failure();
    }

    _log.trace("The Concat at {0} is eliminated", origOp.getLoc());

    for (const auto& keyValue : newSliceOffsetsInputMap) {
        auto origSlice = keyValue.first;
        const auto sliceOffset = keyValue.second.first;
        const auto sliceInput = keyValue.second.second;

        rewriter.setInsertionPoint(origSlice);
        rewriter.replaceOpWithNewOp<VPU::SliceOp>(origSlice, origSlice.getResult().getType(), sliceInput,
                                                  getIntArrayAttr(getContext(), sliceOffset),
                                                  origSlice.getStaticSizes());
    }

    return mlir::success();
}

//
// EliminateSameSiblingConcat
//

/**
 * Optimize the pattern when sibling concat ops have same input type generate by the same root op.
 * Const input of concat should be splat and has the same splat value.
 *
 *                             Op                                       Op
 *                       /           \                                   |
 *         (PermuteCast)            (PermuteCast)                    (PermuteCast)
 *               \    Const1           \      Const2                     |   Const1
 *                \     /               \      /          --->           |   /
 *                Concat1                Concat2                       Concat1
 *                   |                     |                            /   \
 *                  Op1                   Op2                          Op1  Op2
 */
class EliminateSameSiblingConcat final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    EliminateSameSiblingConcat(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        setDebugName("EliminateSameSiblingConcat");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult EliminateSameSiblingConcat::matchAndRewrite(VPU::ConcatOp origOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatOp at loc '{0}'", origOp.getLoc());

    auto getSplatValue = [](Const::DeclareOp constOp) -> std::optional<int64_t> {
        auto content = constOp.getContent();
        if (!content.isSplat()) {
            return std::nullopt;
        }
        return content.getSplatValue<int64_t>();
    };

    SmallVector<std::pair<int64_t, mlir::Value>> concatInputs;
    int64_t numActInput = 0;
    for (const auto& p : origOp.getInputs() | indexed) {
        const auto indexInputPair = std::make_pair(p.index(), p.value());
        const auto inputOp = p.value().getDefiningOp();
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(inputOp)) {
            if (!getSplatValue(constOp).has_value()) {
                _log.trace("Constant input is not splat value");
                return mlir::failure();
            }
            concatInputs.push_back(std::move(indexInputPair));
        } else {
            concatInputs.insert(concatInputs.begin(), std::move(indexInputPair));
            ++numActInput;
        }
    }
    if (numActInput != 1) {
        return mlir::failure();
    }

    // The single activation
    auto root = concatInputs[0].second.getDefiningOp();
    auto prePermuteCastOp = mlir::dyn_cast_or_null<VPU::PermuteCastOp>(root);
    if (prePermuteCastOp) {
        if (!prePermuteCastOp->hasOneUse()) {
            _log.trace("PermuteCast parent has more than one use");
            return mlir::failure();
        }
        root = prePermuteCastOp.getInput().getDefiningOp();
    }

    if (root == nullptr || root->hasOneUse()) {
        _log.trace("Root can't be found or has only one user");
        return mlir::failure();
    }

    auto concatsAreMatched = [&](VPU::ConcatOp concatUserOp) {
        for (auto origInput : concatInputs) {
            const auto userInput = concatUserOp.getOperand(origInput.first);
            if (userInput.getType() != origInput.second.getType()) {
                return false;
            }

            auto userInputOp = userInput.getDefiningOp<Const::DeclareOp>();
            auto origInputOp = origInput.second.getDefiningOp<Const::DeclareOp>();
            if ((userInputOp == nullptr) ^ (origInputOp == nullptr)) {
                return false;
            }
            if (userInputOp == nullptr && origInputOp == nullptr) {
                continue;
            }

            auto splatValue = getSplatValue(userInputOp);
            if (!splatValue.has_value()) {
                return false;
            }

            if (splatValue.value() != getSplatValue(origInputOp).value()) {
                _log.trace("Splat value is not the same as sibling op");
                return false;
            }
        }
        return true;
    };

    const auto origOffsets = origOp.getStaticOffsetsAttr();
    SmallVector<VPU::ConcatOp> concatOps;
    for (auto user : root->getUsers()) {
        if (auto permuteCastUserOp = mlir::dyn_cast<VPU::PermuteCastOp>(user)) {
            if (!permuteCastUserOp->hasOneUse() ||
                (prePermuteCastOp &&
                 permuteCastUserOp.getOutput().getType() != prePermuteCastOp.getOutput().getType())) {
                continue;
            }
            user = *permuteCastUserOp->getUsers().begin();
        }

        auto concatUserOp = mlir::dyn_cast<VPU::ConcatOp>(user);
        if (concatUserOp == nullptr || concatUserOp == origOp) {
            continue;
        }
        if (concatUserOp.getInputs().size() != origOp.getInputs().size() ||
            concatUserOp.getOutput().getType() != origOp.getOutput().getType()) {
            continue;
        }

        const auto userOffsets = concatUserOp.getStaticOffsetsAttr();
        if (userOffsets != origOffsets) {
            continue;
        }
        if (!concatsAreMatched(concatUserOp)) {
            continue;
        }

        concatOps.push_back(concatUserOp);
    }

    if (concatOps.empty()) {
        _log.trace("Same sibling concat ops can't be found");
        return mlir::failure();
    }

    for (auto concatOp : concatOps) {
        rewriter.replaceOp(concatOp, origOp.getOutput());
        _log.trace("The Concat at {0} is eliminated", concatOp.getLoc());
    }

    return mlir::success();
}

//
// OptimizeConcatPass
//

class OptimizeConcatPass final : public VPU::impl::OptimizeConcatBase<OptimizeConcatPass> {
public:
    explicit OptimizeConcatPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

void OptimizeConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<EliminateConcat>(&ctx, _log);
    patterns.insert<EliminateSameSiblingConcat>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeConcatPass(Logger log) {
    return std::make_unique<OptimizeConcatPass>(log);
}
