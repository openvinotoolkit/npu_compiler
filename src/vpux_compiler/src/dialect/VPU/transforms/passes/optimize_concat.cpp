//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/utils/slice_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
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
    VPU::PermuteCastOp permuteCastOp = nullptr;
    mlir::Operation* sliceParentOp = origOp;
    if (origOp->hasOneUse()) {
        permuteCastOp = mlir::dyn_cast_or_null<VPU::PermuteCastOp>(*origOp->getUsers().begin());
        if (permuteCastOp != nullptr) {
            if (getShape(permuteCastOp.getInput()) != getShape(permuteCastOp.getOutput())) {
                return mlir::failure();
            }
            sliceParentOp = permuteCastOp;
        }
    }

    const auto allUsersSliceSubTensors = llvm::all_of(sliceParentOp->getUsers(), [&](auto userOp) {
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
        auto sliceInput = keyValue.second.second;

        rewriter.setInsertionPoint(origSlice);
        if (permuteCastOp != nullptr) {
            sliceInput = VPU::PermuteCastOp::create(rewriter, permuteCastOp.getLoc(), sliceInput,
                                                    permuteCastOp.getDstOrderAttr(), permuteCastOp.getMemPermAttr());
        }

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
        const auto concatOpLoc = concatOp.getLoc();
        rewriter.replaceOp(concatOp, origOp.getOutput());
        _log.trace("The Concat at {0} is eliminated", concatOpLoc);
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

namespace {

bool isConsistentReshape(VPU::AffineReshapeOp reshapeOp1, VPU::AffineReshapeOp reshapeOp2) {
    return llvm::equal(reshapeOp1.getDimMapping(), reshapeOp2.getDimMapping()) &&
           llvm::equal(reshapeOp1.getShapeValue(), reshapeOp2.getShapeValue());
}

bool isGatherDMA(mlir::Value input) {
    return mlir::isa_and_nonnull<VPU::GatherDMAOp>(input.getDefiningOp());
};

bool remapOffsetsToReshapeInput(VPU::AffineReshapeOp& reshapeOp, SmallVector<SmallVector<int64_t>>& origOpOffsets,
                                const Logger& log) {
    const auto outShape = parseIntArrayAttr<int64_t>(reshapeOp.getShapeValue());
    const auto inShape = getShape(reshapeOp.getInput());
    for (auto ind : irange(origOpOffsets.size())) {
        if (inShape.size() > outShape.size() || outShape.size() - inShape.size() > 1) {
            return false;
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
        for (auto dim : parseIntArrayOfArrayAttr<int64_t>(reshapeOp.getDimMapping())) {
            switch (dim.size()) {
            case 1:
                newOrigOpOffsets.push_back(origOpOffsets[ind][dim[0]]);
                break;
            case 2:
                newOrigOpOffsets.push_back(origOpOffsets[ind][dim[0]] * outShape[dim[1]] + origOpOffsets[ind][dim[1]]);
                break;
            default: {
                log.trace("Unsupported AffineReshape operation : {0}", reshapeOp);
                return false;
            }
            }
        }
        origOpOffsets[ind] = std::move(newOrigOpOffsets);
    }
    return true;
}

}  // namespace

mlir::LogicalResult FuseMultipleConcatOpsAroundGatherDMA::matchAndRewrite(VPU::ConcatOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatOp at loc '{0}'", origOp.getLoc());
    SmallVector<VPU::ConcatOp> parentConcatOps;
    std::optional<VPU::AffineReshapeOp> reshapeOp;

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
                if (reshapeOp.has_value() && !isConsistentReshape(reshape, reshapeOp.value())) {
                    return false;
                }
                parentConcatOps.push_back(concat);
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

    if (reshapeOp.has_value() && !remapOffsetsToReshapeInput(reshapeOp.value(), origOpOffsets, _log)) {
        return mlir::failure();
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

    auto newConcat = VPU::ConcatOp::create(rewriter, origOp.getLoc(), newInputs, /*per_axis=*/nullptr,
                                           getIntArrayOfArray(rewriter.getContext(), newConcatOffsets));

    if (reshapeOp.has_value()) {
        const auto outShape = getShape(origOp.getOutput()).toValues();
        rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(origOp, newConcat.getOutput(), reshapeOp->getDimMapping(),
                                                          getIntArrayAttr(rewriter.getContext(), outShape));
        return mlir::success();
    }

    rewriter.replaceOp(origOp, newConcat);
    return mlir::success();
}

//
// AbsorbConcatIntoForOpAroundGatherDMA
//

/**
 * Absorbs a VPU.Concat whose inputs are produced by independent `scf.for` loops directly instead of concatenating the
 * results afterwards. A single `tensor.empty()` is created whose shape matches the Concat's output. The loops then use
 * this bigger tensor as its iter_arg, with `tensor.insert_slice` destination offset shifted by its position in the
 * Concat.
 * This is intended as a generalized replacement for the `FuseMultipleConcatOpsAroundGatherDMA` if DMA tiling is done
 * with `scf.for` loops.
 */
class AbsorbConcatIntoForOpAroundGatherDMA final : public mlir::OpRewritePattern<VPU::ConcatOp> {
public:
    AbsorbConcatIntoForOpAroundGatherDMA(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::ConcatOp>(ctx), _log(log) {
        setDebugName("AbsorbConcatIntoForOpAroundGatherDMA");
    }

    mlir::LogicalResult matchAndRewrite(VPU::ConcatOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

namespace {

bool canAbsorbConcat(mlir::scf::ForOp forOp) {
    // We only support 1D tiling: each loop tiles along one axis and maps to one Concat input. 2D tiling (nested loops
    // producing one Concat input, possibly with a strided concat) is not handled: the outer loop's iter_arg feeds the
    // nested loop, so the "iter_arg used only by this loop's own insert_slice" check below rejects it, and the offsets
    // are not accumulated across loops. Revisit if such patterns are generated.
    if (forOp->getNumResults() != 1 || !forOp->hasOneUse()) {
        return false;
    }
    const auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(forOp.getResult(0).getType());
    if (resultType == nullptr || !resultType.hasStaticShape()) {
        return false;
    }
    if (auto initArg = forOp.getInitArgs().front()) {
        if (!mlir::isa_and_nonnull<mlir::tensor::EmptyOp>(initArg.getDefiningOp())) {
            return false;
        }
    }

    // The loop's iter_arg must have at least one use.
    auto uses = forOp.getRegionIterArg(0).getUses();
    if (uses.empty()) {
        return false;
    }

    // iter_arg must only be used by the loop's own `tensor.insert_slice` in the body, and that slice must come
    // from a GatherDMA, matching FuseMultipleConcatOpsAroundGatherDMA
    auto* bodyBlock = forOp.getBody();
    for (auto& use : uses) {
        auto insertSlice = mlir::dyn_cast_or_null<mlir::tensor::InsertSliceOp>(use.getOwner());
        if (insertSlice == nullptr || insertSlice->getBlock() != bodyBlock ||
            insertSlice.getDest() != forOp.getRegionIterArg(0) || !isGatherDMA(insertSlice.getSource())) {
            return false;
        }
    }
    return true;
};

bool tryCollectForOpInput(mlir::Operation* op, SmallVector<mlir::scf::ForOp>& parentForOps,
                          std::optional<VPU::AffineReshapeOp>& reshapeOp, bool& sawDirectForOpInput) {
    if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(op)) {
        // A direct (non-reshaped) input cannot be mixed with reshaped ones
        if (reshapeOp.has_value()) {
            return false;
        }
        if (canAbsorbConcat(forOp)) {
            parentForOps.push_back(forOp);
            sawDirectForOpInput = true;
            return true;
        }
    }
    // Handle reshape ops.
    auto reshape = mlir::dyn_cast_or_null<VPU::AffineReshapeOp>(op);
    if (!reshape) {
        return false;
    }
    if (sawDirectForOpInput || !reshape->hasOneUse()) {
        return false;
    }
    if (auto forOp = mlir::dyn_cast_or_null<mlir::scf::ForOp>(reshape.getInput().getDefiningOp())) {
        if (reshapeOp.has_value() && !isConsistentReshape(reshapeOp.value(), reshape)) {
            return false;
        }
        if (canAbsorbConcat(forOp)) {
            parentForOps.push_back(forOp);
            reshapeOp = reshape;
            return true;
        }
    }
    return false;
};
}  // namespace

mlir::LogicalResult AbsorbConcatIntoForOpAroundGatherDMA::matchAndRewrite(VPU::ConcatOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got ConcatOp at loc '{0}'", origOp.getLoc());
    SmallVector<mlir::scf::ForOp> parentForOps;
    std::optional<VPU::AffineReshapeOp> reshapeOp;
    bool sawDirectForOpInput = false;

    const auto outputType = mlir::cast<NDTypeInterface>(origOp.getOutput().getType());

    // We only support static shapes for now.
    if (!outputType.getShape().isStatic()) {
        return mlir::failure();
    }

    for (auto input : origOp.getInputs()) {
        if (!tryCollectForOpInput(input.getDefiningOp(), parentForOps, reshapeOp, sawDirectForOpInput)) {
            return mlir::failure();
        }
    }

    if (parentForOps.empty()) {
        return mlir::failure();
    }

    if (!origOp.getStaticOffsets().has_value() || origOp.getStrides().has_value()) {
        _log.error("ConcatOp at {0} has no static offsets or has strides, cannot absorb into loops", origOp.getLoc());
        return mlir::failure();
    }

    SmallVector<SmallVector<int64_t>> concatOffsets{
            parseIntArrayOfArrayAttr<int64_t>(origOp.getStaticOffsets().value())};

    if (reshapeOp.has_value() && !remapOffsetsToReshapeInput(reshapeOp.value(), concatOffsets, _log)) {
        return mlir::failure();
    }

    if (parentForOps.size() != concatOffsets.size()) {
        return mlir::failure();
    }

    // Make sure all loops' outputs have the same element type as the Concat's output.
    const auto hasMismatchedElementTypes = llvm::any_of(parentForOps, [&](auto forOp) {
        return mlir::cast<NDTypeInterface>(forOp.getResult(0).getType()).getElementType() !=
               outputType.getElementType();
    });

    if (hasMismatchedElementTypes) {
        _log.error("Not all loops' outputs have the same element type as the Concat's output");
        return mlir::failure();
    }

    NDTypeInterface sharedDestType;
    // If reshapeOp is present, set the shared destination type to the maximum shape of all the loops' outputs, shifted
    // by their offsets
    if (reshapeOp.has_value()) {
        sharedDestType = mlir::cast<NDTypeInterface>(parentForOps.front().getResult(0).getType());
        Shape sharedDestShape(sharedDestType.getShape());
        for (auto idx : irange(parentForOps.size())) {
            const auto nativeShape = getShape(parentForOps[idx].getResult(0));
            for (auto dim : irange(sharedDestShape.size())) {
                const Dim d(dim);
                sharedDestShape[d] = std::max(sharedDestShape[d], concatOffsets[idx][dim] + nativeShape[d]);
            }
        }
        sharedDestType = sharedDestType.changeShape(ShapeRef(sharedDestShape));
    } else {
        sharedDestType = outputType;
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(sharedDestType);
    if (resultType == nullptr) {
        return mlir::failure();
    }

    rewriter.setInsertionPoint(origOp);
    mlir::Value fusedResult =
            mlir::tensor::EmptyOp::create(rewriter, origOp.getLoc(), resultType, /*dynamicSizes*/ mlir::ValueRange{})
                    .getResult();

    for (auto [ind, forOp] : llvm::enumerate(parentForOps)) {
        auto newForOp = mlir::scf::ForOp::create(rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
                                                 forOp.getStep(), fusedResult);
        for (auto attr : forOp->getAttrs()) {
            newForOp->setAttr(attr.getName(), attr.getValue());
        }
        // Map original induction variables and operands to those of the fused loop.
        mlir::IRMapping mapping;
        mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
        mapping.map(forOp.getRegionIterArgs(), newForOp.getRegionIterArgs());

        rewriter.setInsertionPointToStart(newForOp.getBody());

        // Clone the original loop body, except for insert_slice ops.
        for (mlir::Operation& op : forOp.getBody()->getOperations()) {
            auto insertSlice = mlir::dyn_cast<mlir::tensor::InsertSliceOp>(op);
            if (!insertSlice || insertSlice.getDest() != forOp.getRegionIterArg(0)) {
                rewriter.clone(op, mapping);
                continue;
            }
            auto newOffsets = insertSlice.getMixedOffsets();
            auto newSizes = insertSlice.getMixedSizes();
            auto newStrides = insertSlice.getMixedStrides();
            for (auto* ofrList : {&newOffsets, &newSizes, &newStrides}) {
                for (auto& ofr : *ofrList) {
                    if (auto val = ofr.dyn_cast<mlir::Value>()) {
                        ofr = mapping.lookupOrDefault(val);
                    }
                }
            }
            // Shift every offset by this loop's position within the Concat along that dim and rebuild the op
            for (const auto dim : irange(newOffsets.size())) {
                const int64_t concatOffset = concatOffsets[ind][dim];
                auto shiftMap =
                        mlir::AffineMap::get(1, 0, mlir::getAffineDimExpr(0, rewriter.getContext()) + concatOffset);
                newOffsets[dim] = mlir::affine::makeComposedFoldedAffineApply(rewriter, insertSlice.getLoc(), shiftMap,
                                                                              {newOffsets[dim]});
            }

            auto newInsertSlice = mlir::tensor::InsertSliceOp::create(
                    rewriter, insertSlice.getLoc(), mapping.lookupOrDefault(insertSlice.getSource()),
                    mapping.lookupOrDefault(insertSlice.getDest()), newOffsets, newSizes, newStrides);

            mapping.map(insertSlice.getResult(), newInsertSlice.getResult());
        }

        fusedResult = newForOp.getResult(0);
        rewriter.setInsertionPoint(origOp);
    }

    if (reshapeOp.has_value()) {
        const auto outShape = getShape(origOp.getOutput()).toValues();
        rewriter.replaceOpWithNewOp<VPU::AffineReshapeOp>(origOp, fusedResult, reshapeOp->getDimMapping(),
                                                          getIntArrayAttr(rewriter.getContext(), outShape));
        return mlir::success();
    }

    // Replace ConcatOp with the new loops' results
    rewriter.replaceOp(origOp, fusedResult);

    return mlir::success();
}

//
// OptimizeConcatPass
//

class OptimizeConcatPass final : public VPU::impl::OptimizeConcatBase<OptimizeConcatPass> {
public:
    explicit OptimizeConcatPass(bool optimizeOnlyOuterConcat, Logger log) {
        Base::initLogger(log, Base::getArgumentName());
        _optimizeOnlyOuterConcat = optimizeOnlyOuterConcat;
    }

private:
    mlir::LogicalResult initializeOptions(
            StringRef options, llvm::function_ref<mlir::LogicalResult(const llvm::Twine&)> errorHandler) final;
    void safeRunOnFunc() final;
    bool _optimizeOnlyOuterConcat = false;
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
    return mlir::success();
}

void OptimizeConcatPass::safeRunOnFunc() {
    auto func = getOperation();
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.insert<EliminateConcat>(&ctx, _log, _optimizeOnlyOuterConcat);
    patterns.insert<EliminateSameSiblingConcat>(&ctx, _log);
    patterns.insert<FuseMultipleConcatOpsAroundGatherDMA>(&ctx, _log);
    patterns.insert<AbsorbConcatIntoForOpAroundGatherDMA>(&ctx, _log);

    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns), vpux::getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createOptimizeConcatPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createOptimizeConcatPass(bool optimizeOnlyOuterConcat, Logger log) {
    return std::make_unique<OptimizeConcatPass>(optimizeOnlyOuterConcat, log);
}
