//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/net/network_info_utils.hpp"
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
    EliminateConcat(mlir::MLIRContext* ctx, Logger log, bool optimizeOnlyOuterConcat)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log), _optimizeOnlyOuterConcat(optimizeOnlyOuterConcat) {
        setDebugName("EliminateConcat");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _optimizeOnlyOuterConcat;
};

mlir::LogicalResult EliminateConcat::matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const {
    if (!origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }

    const auto concatOffsets = parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsets().value());
    DenseMap<VPU::SliceOp, std::pair<SmallVector<int64_t>, mlir::Value>> newSliceOffsetsInputMap;
    auto concatOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());

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

            auto axes = getConcatAxes(origOp);
            auto dim = getHighestNonTrivialDim(concatOutputType.getShape(), concatOutputType.getDimsOrder())
                               .value_or(Dim(0));
            // If slice axis is not the highest non-trivial dim in layout, the concat optimization will produce
            // non-inplace slice ops, the spilling can not be avoided. Thus we prevent it.
            if (_optimizeOnlyOuterConcat && ((axes.size() > 1) || (dim.ind() != (*axes.begin())))) {
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

    SmallVector<std::pair<int64_t, mlir::Value>> concatInputs;
    int64_t numActInput = 0;
    for (const auto& p : origOp.getInputs() | indexed) {
        const auto indexInputPair = std::make_pair(p.index(), p.value());
        const auto inputOp = p.value().getDefiningOp();
        if (auto constOp = mlir::dyn_cast_or_null<Const::DeclareOp>(inputOp)) {
            if (!constOp.getContentAttr().isSplat()) {
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

            auto splatValue = Const::getSplatValue<int64_t>(userInputOp);
            if (mlir::failed(splatValue)) {
                return false;
            }

            if (splatValue.value() != Const::getSplatValue<int64_t>(origInputOp).value()) {
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
// FuseMultipleConcatOpsAroundGatherDMA
//

/**
 * The Gather DMA operation is tiled by two separate passes. The first one is TileGather in order to satisfy following
 * limitation : "the input size after axis should be less than 4096". The second one is inside the usual tiling
 * pipeline. Those will generate multiple Concat operations that will be fused here.
 */
class FuseMultipleConcatOpsAroundGatherDMA final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    FuseMultipleConcatOpsAroundGatherDMA(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        setDebugName("FuseMultipleConcatOpsAroundGatherDMA");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseMultipleConcatOpsAroundGatherDMA::matchAndRewrite(VPU::ConcatOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatOp at loc '{0}'", origOp.getLoc());
    SmallVector<VPU::ConcatOp> parentConcatOps;
    mlir::ArrayAttr dimMapping;
    std::optional<VPU::AffineReshapeOp> reshapeOp;

    auto isGatherDMA = [](auto input) {
        return mlir::isa_and_nonnull<VPU::GatherDMAOp>(input.getDefiningOp());
    };

    auto hasAnotherConcatAsIn = [&](mlir::Operation* op) {
        if (auto concat = mlir::dyn_cast_or_null<VPU::ConcatOp>(op)) {
            if (!llvm::all_of(concat.getInputs(), isGatherDMA)) {
                return false;
            }

            parentConcatOps.push_back(concat);
            return true;
        }
        if (auto reshape = mlir::dyn_cast_or_null<VPU::AffineReshapeOp>(op)) {
            if (auto concat = mlir::dyn_cast_or_null<VPU::ConcatOp>(reshape.getInput().getDefiningOp())) {
                if (!llvm::all_of(concat.getInputs(), isGatherDMA)) {
                    return false;
                }
                if (dimMapping != nullptr && !llvm::equal(dimMapping, reshape.getDimMapping())) {
                    return false;
                }
                parentConcatOps.push_back(concat);
                dimMapping = reshape.getDimMapping();
                reshapeOp = reshape;
                return true;
            }
        }
        return false;
    };

    for (auto input : origOp.getInputs()) {
        if (!hasAnotherConcatAsIn(input.getDefiningOp())) {
            return mlir::failure();
        }
    }

    if (parentConcatOps.size() < 2) {
        return mlir::failure();
    }

    if (!origOp.getStaticOffsets().has_value()) {
        return mlir::failure();
    }

    SmallVector<mlir::Value> newInputs;
    SmallVector<SmallVector<int64_t>> newConcatOffsets;
    SmallVector<SmallVector<int64_t>> origOpOffsets{
            parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsets().value())};

    if (reshapeOp.has_value()) {
        for (auto ind : irange(origOpOffsets.size())) {
            auto outShape = parseIntArrayAttr<int64_t>(reshapeOp.value().getShapeValue());
            if (outShape.size() -
                        mlir::dyn_cast<NDTypeInterface>(reshapeOp.value().getInput().getType()).getShape().size() >
                1) {
                return mlir::failure();
            }

            SmallVector<int64_t> newOrigOpOffsets;
            // If there is a reshape op between orig Concat and parent Concat we need to recalculate the offsets,
            // because optimized Concat will have direct GatherDMA inputs and AffineReshape inserted after.
            // Ex:
            //     GatherDMA 2D     GatherDMA 2D   GatherDMA 2D
            //              \           |          /
            //                       Concat 2D - Offsets : [0, 0], [0, 2], [0, 4]
            //                          |
            //                AffineReshape 2D -> 3D
            //                          |      ..... / ...... / (Here will be same multiple subgraphs)
            //                      Concat 3D  - Offsets : [0, 0, 0], ....
            //   ->
            //     GatherDMA 2D     GatherDMA 2D   GatherDMA 2D
            //              \           |          /
            //                       Concat 2D - Offsets : [0, 0], [0, 2], [0, 4] ....
            //                          |
            //                AffineReshape 2D -> 3D
            for (auto dim : parseIntArrayOfArrayAttr<int64_t>(dimMapping)) {
                switch (dim.size()) {
                case 1:
                    newOrigOpOffsets.push_back(origOpOffsets[ind][dim[0]]);
                    break;
                case 2:
                    newOrigOpOffsets.push_back(origOpOffsets[ind][dim[0]] * outShape[dim[1]] +
                                               origOpOffsets[ind][dim[1]]);
                    break;
                default: {
                    _log.trace("Unsupported AffineReshape operation : {0}", reshapeOp);
                    return mlir::failure();
                }
                }
            }
            origOpOffsets[ind] = std::move(newOrigOpOffsets);
        }
    }

    for (auto idx : irange(parentConcatOps.size())) {
        auto offsets = parseIntArrayOfArrayAttr<int64_t>(parentConcatOps[idx].getStaticOffsets().value());
        for (size_t i = 0; i < parentConcatOps[idx].getInputs().size(); i++) {
            newInputs.push_back(parentConcatOps[idx].getInputs()[i]);
            SmallVector<int64_t> newOffset;
            newOffset.reserve(offsets[i].size());
            for (auto ind : irange(offsets[i].size())) {
                newOffset.push_back(origOpOffsets[idx][ind] + offsets[i][ind]);
            }

            newConcatOffsets.push_back(newOffset);
        }
    }

    auto newConcat = rewriter.create<VPU::ConcatOp>(origOp.getLoc(), newInputs, /*per_axis=*/nullptr,
                                                    getIntArrayOfArray(rewriter.getContext(), newConcatOffsets));

    if (reshapeOp.has_value()) {
        const auto outShape = getShape(origOp.getOutput()).toValues();
        rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(origOp, newConcat.getOutput(), dimMapping,
                                                          getIntArrayAttr(rewriter.getContext(), outShape));
        return mlir::success();
    }

    rewriter.replaceOp(origOp, newConcat);
    return mlir::success();
}

//
// OptimizeConcatPass
//

class OptimizeConcatPass final : public VPU::impl::OptimizeConcatBase<OptimizeConcatPass> {
public:
    explicit OptimizeConcatPass(bool optimizeOnlyOuterConcat, bool disablePassOnEntryFunction, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        _optimizeOnlyOuterConcat = optimizeOnlyOuterConcat;
        _disablePassOnEntryFunction = disablePassOnEntryFunction;
    }

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnFunc() final;
    bool _optimizeOnlyOuterConcat = false;
    bool _disablePassOnEntryFunction = false;
};

mlir::LogicalResult OptimizeConcatPass::initializeOptions(
        StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) {
    if (mlir::failed(Base::initializeOptions(options, errorHandler))) {
        return mlir::failure();
    }
    if (optimizeOnlyOuterConcat.hasValue()) {
        _log.trace("Overloading optimizeOnlyOuterConcat with an MLIR variable {0}", optimizeOnlyOuterConcat.getValue());
        _optimizeOnlyOuterConcat = optimizeOnlyOuterConcat.getValue();
    }

    if (disablePassOnEntryFunction.hasValue()) {
        _log.trace("Overloading disablePassOnEntryFunction with an MLIR variable {0}",
                   disablePassOnEntryFunction.getValue());
        _disablePassOnEntryFunction = disablePassOnEntryFunction.getValue();
    }
    return mlir::success();
}

void OptimizeConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    auto entryPointFunc = vpux::net::findEntryPointFunc(func, _log);
    if (_disablePassOnEntryFunction && (func == entryPointFunc)) {
        _log.trace("Skipping function {0} in HostCompile mode", func.getName());
        return;
    }

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<EliminateConcat>(&ctx, _log, _optimizeOnlyOuterConcat);
    patterns.insert<EliminateSameSiblingConcat>(&ctx, _log);
    patterns.insert<FuseMultipleConcatOpsAroundGatherDMA>(&ctx, _log);

    if (mlir::failed(
                mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeConcatPass(bool optimizeOnlyOuterConcat,
                                                                bool disablePassOnEntryFunction, Logger log) {
    return std::make_unique<OptimizeConcatPass>(optimizeOnlyOuterConcat, disablePassOnEntryFunction, log);
}
