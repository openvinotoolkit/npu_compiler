//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/pooling_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MOVEPERMUTEPOSTELTWISE
#define GEN_PASS_DEF_MOVEPERMUTEPOSTELTWISE
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;
using namespace IE;

namespace {

using VerifyCb = FuncRef<bool(mlir::Operation*)>;

//
// MovePermutePostEltwisePass
//

bool isEltwiseGroupConvolution(IE::GroupConvolutionOp groupConvOp) {
    const auto kernelSize = getShape(groupConvOp.getFilter());
    if (kernelSize[Dims4D::Filter::KX] != 1 || kernelSize[Dims4D::Filter::KY] != 1) {
        return false;
    }

    const auto hasLargeDim = [](const int64_t val) -> bool {
        return val != 1;
    };
    const auto hasPadding = [](const int64_t val) -> bool {
        return val != 0;
    };

    const auto strides = parseIntArrayAttr<int64_t>(groupConvOp.getStrides());
    if (std::any_of(strides.begin(), strides.end(), hasLargeDim)) {
        return false;
    }

    const auto dilations = parseIntArrayAttr<int64_t>(groupConvOp.getDilations());
    if (std::any_of(dilations.begin(), dilations.end(), hasLargeDim)) {
        return false;
    }

    const auto padsBegin = parseIntArrayAttr<int64_t>(groupConvOp.getPadsBegin());
    if (std::any_of(padsBegin.begin(), padsBegin.end(), hasPadding)) {
        return false;
    }

    const auto padsEnd = parseIntArrayAttr<int64_t>(groupConvOp.getPadsEnd());
    return !std::any_of(padsEnd.begin(), padsEnd.end(), hasPadding);
}

class MovePermutePostEltwisePass final : public IE::impl::MovePermutePostEltwiseBase<MovePermutePostEltwisePass> {
public:
    MovePermutePostEltwisePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// PermuteEltwiseRewriter
//

template <class EltwiseOp>
class PermuteEltwiseRewriter final : public mlir::OpRewritePattern<EltwiseOp> {
public:
    PermuteEltwiseRewriter(mlir::MLIRContext* ctx, VerifyCb verifyFunc, size_t numInputs, Logger log)
            : mlir::OpRewritePattern<EltwiseOp>(ctx), _verifyFunc(verifyFunc), _numInputs(numInputs), _log(log) {
        this->setDebugName("PermuteEltwiseRewriter");
    }

public:
    mlir::LogicalResult matchAndRewrite(EltwiseOp eltwiseOp, mlir::PatternRewriter& rewriter) const final;

private:
    VerifyCb _verifyFunc;
    size_t _numInputs;
    Logger _log;
};

// Get the mapped Shape of the mapping from the source layout to the result + permutation layout
// with no data change
// e.g., source layout NCHW, shape is [n, c, h, w], mem_perm layout is NWCH
// return layout NHWC, the mapped result shape should be [n, h, w, c]
Shape getMappedShape(DimsOrder sourceLayout, DimsOrder resultLayout, DimsOrder memPermLayout, ShapeRef sourceShape) {
    auto resultShape = Shape(sourceShape.size(), 1);
    auto mappedShape = resultShape;
    auto sourcePerm = sourceLayout.toPermutation();
    auto resultPerm = resultLayout.toPermutation();
    auto memPerm = memPermLayout.toPermutation();
    for (auto dim : irange(sourceShape.size())) {
        resultShape[resultPerm[dim]] = sourceShape[sourcePerm[dim]];
    }
    for (auto dim : irange(mappedShape.size())) {
        mappedShape[memPerm[dim]] = resultShape[resultPerm[dim]];
    }
    return mappedShape;
}

mlir::Operation* getEltwiseInputPermute(mlir::Value eltwiseInput) {
    auto parentOp = eltwiseInput.getDefiningOp();
    while (parentOp) {
        if (auto parentPermute = mlir::dyn_cast<IE::MemPermuteOp>(parentOp)) {
            // Case when there is a MemPermute and the processing of EltWise above adds another MemPermute
            // Skipping now, as canonicalizer takes care of optimizing both MemPermutes
            auto grandParentOp = parentPermute.getInput().getDefiningOp();
            if (mlir::isa_and_nonnull<IE::MemPermuteOp>(grandParentOp)) {
                auto grandParentPermute = mlir::cast<IE::MemPermuteOp>(grandParentOp);
                if (parentPermute.getDstOrder() == grandParentPermute.getDstOrder()) {
                    return nullptr;
                }
            }
            return parentPermute.getOperation();
        } else if (auto parentPermute = mlir::dyn_cast<IE::PermuteQuantizeOp>(parentOp)) {
            // Case when there is a PermuteQuantize and the processing of EltWise above adds another PermuteQuantize
            // Skipping now, as canonicalizer takes care of optimizing both PermuteQuantizes
            auto grandParentOp = parentPermute.getInput().getDefiningOp();
            if (mlir::isa_and_nonnull<IE::PermuteQuantizeOp>(grandParentOp)) {
                auto grandParentPermute = mlir::cast<IE::PermuteQuantizeOp>(grandParentOp);
                if (parentPermute.getDstOrder() == grandParentPermute.getDstOrder()) {
                    return nullptr;
                }
            }
            // Skipping PermuteQuantize which also performs padding for next NCE Eltwise
            const auto isZero = [](const int64_t val) -> bool {
                return val == 0;
            };
            const auto padsBegin = parseIntArrayAttr<int64_t>(parentPermute.getPadsBegin());
            const auto padsEnd = parseIntArrayAttr<int64_t>(parentPermute.getPadsEnd());
            if (!(llvm::all_of(padsBegin, isZero) && llvm::all_of(padsEnd, isZero))) {
                return nullptr;
            }
            return parentPermute.getOperation();
        } else if (auto parentQuantizeCast = mlir::dyn_cast<IE::QuantizeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentQuantizeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentQuantizeCast.getInput().getDefiningOp();
            continue;
        } else if (auto parentShapeCast = mlir::dyn_cast<IE::ShapeCastOp>(parentOp)) {
            if (VPU::hasMultiBranches(parentShapeCast.getOperation())) {
                return nullptr;
            }
            parentOp = parentShapeCast.getSource().getDefiningOp();
            continue;
        } else {
            return nullptr;
        }
    }
    return nullptr;
}

SmallVector<mlir::Operation*> getPermutesToMove(ArrayRef<mlir::Operation*> permutes) {
    if (permutes.size() == 1) {
        return SmallVector<mlir::Operation*>({permutes[0]});
    }
    if (permutes.size() == 2) {
        if (permutes[0] == permutes[1]) {
            return SmallVector<mlir::Operation*>({permutes[0]});
        } else {
            return SmallVector<mlir::Operation*>({permutes[0], permutes[1]});
        }
    }
    VPUX_THROW("getPermutesToMove: Unsupported number of elements. Expected 1 or 2, got {0}", permutes.size());
}

bool isSplatConstant(Const::DeclareOp constOp) {
    return (constOp != nullptr) && constOp.getContentAttr().isSplat();
}

/* Rewrite the pattern from:

   Permute      Permute
      |          |
(QuantizeCast) (QuantizeCast)
       \        /
         Eltwise
            |
      (QuantizeCast)
           ...

    to:
PermuteCast  PermuteCast
      |          |
 ShapeCast   ShapeCast
      |          |
(QuantizeCast) (QuantizeCast)
       \        /
         Eltwise
            |
     (QuantizeCast)
            |
        ShapeCast
            |
       PermuteCast
            |
         Permute
            |
           ...
 */
template <class EltwiseOp>
mlir::LogicalResult PermuteEltwiseRewriter<EltwiseOp>::matchAndRewrite(EltwiseOp eltwiseOp,
                                                                       mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got '{1}' at '{2}'", this->getDebugName(), eltwiseOp->getName(), eltwiseOp->getLoc());
    auto ctx = this->getContext();
    const auto hasEltwiseTrait = eltwiseOp->template hasTrait<IE::EltwiseOp>();
    if (!(hasEltwiseTrait || (_verifyFunc && _verifyFunc(eltwiseOp.getOperation())))) {
        return mlir::failure();
    }

    SmallVector<mlir::Operation*> inputPermutes;
    for (size_t inIdx = 0; inIdx < _numInputs; inIdx++) {
        inputPermutes.push_back(getEltwiseInputPermute(eltwiseOp->getOperand(inIdx)));
    }
    const auto isPermute = [](const mlir::Operation* op) -> bool {
        return mlir::isa_and_nonnull<IE::MemPermuteOp, IE::PermuteQuantizeOp>(op);
    };
    auto allInputsArePermutes = std::all_of(inputPermutes.begin(), inputPermutes.end(), isPermute);
    if (!allInputsArePermutes) {
        return mlir::failure();
    }
    SmallVector<vpux::NDTypeInterface> permuteInputTypes;
    const auto getInputType = [](mlir::Operation* permute) -> vpux::NDTypeInterface {
        return permute->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
    };
    std::transform(inputPermutes.begin(), inputPermutes.end(), std::back_inserter(permuteInputTypes), getInputType);

    SmallVector<DimsOrder> permuteInputLayouts;
    const auto getInputLayout = [](mlir::Operation* permute) -> DimsOrder {
        return DimsOrder::fromValue(permute->getOperand(0));
    };
    std::transform(inputPermutes.begin(), inputPermutes.end(), std::back_inserter(permuteInputLayouts), getInputLayout);

    SmallVector<DimsOrder> permuteMemPermLayouts;
    const auto getMemPermLayout = [](mlir::Operation* op) -> DimsOrder {
        if (auto permute = mlir::dyn_cast<IE::MemPermuteOp>(op)) {
            return DimsOrder::fromAffineMap(permute.getMemPermAttr().getValue());
        } else if (auto permute = mlir::dyn_cast<IE::PermuteQuantizeOp>(op)) {
            return DimsOrder::fromAffineMap(permute.getMemPermAttr().getValue());
        } else {
            VPUX_THROW("Unsupported operation type, got {0}", op->getLoc());
        }
    };
    std::transform(inputPermutes.begin(), inputPermutes.end(), std::back_inserter(permuteMemPermLayouts),
                   getMemPermLayout);

    auto eltwiseOutElemType = eltwiseOp.getOutput().getType().template cast<vpux::NDTypeInterface>().getElementType();
    SmallVector<bool> isPermuteElemTypeEquals;
    const auto getElemTypeEqual = [eltwiseOutElemType](mlir::Operation* op) -> bool {
        if (mlir::isa<IE::MemPermuteOp>(op)) {
            return true;
        }
        auto srcElemType = op->getOperand(0).getType().cast<NDTypeInterface>().getElementType();
        auto dstElemType = op->getResult(0).getType().cast<NDTypeInterface>().getElementType();
        return (srcElemType == dstElemType) && (eltwiseOutElemType.isF16() || eltwiseOutElemType.isF32());
    };
    std::transform(inputPermutes.begin(), inputPermutes.end(), std::back_inserter(isPermuteElemTypeEquals),
                   getElemTypeEqual);

    auto eltwiseInput1Type = eltwiseOp->getOperand(0).getType().template cast<vpux::NDTypeInterface>();
    auto eltwiseInputLayout = eltwiseInput1Type.getDimsOrder();
    auto eltwiseInputShape = eltwiseInput1Type.getShape();
    auto eltwiseOutputLayout = eltwiseOp.getOutput().getType().template cast<vpux::NDTypeInterface>().getDimsOrder();

    const auto patternCanBeConverted = [&]() -> bool {
        const auto firstInputLayout = permuteInputLayouts[0];
        const auto isSameLayout = [firstInputLayout](const DimsOrder layout) -> bool {
            return firstInputLayout == layout;
        };
        if (!std::all_of(permuteInputLayouts.begin(), permuteInputLayouts.end(), isSameLayout)) {
            // If the two inputs before Permutes have different layouts, the case is not supported
            // e.g., input1 layout is NCHW, input2 layout is NWHC
            return false;
        }
        if (eltwiseInputLayout != DimsOrder::NHWC) {
            // Consider the layout is adjusted to NHWC for Eltwise
            return false;
        }
        if (eltwiseOutputLayout != eltwiseInputLayout) {
            return false;
        }
        const auto firstMemPermLayout = permuteMemPermLayouts[0];
        const auto isSameMemPerm = [firstMemPermLayout](const DimsOrder layout) -> bool {
            return firstMemPermLayout == layout;
        };
        if (!std::all_of(permuteMemPermLayouts.begin(), permuteMemPermLayouts.end(), isSameMemPerm)) {
            return false;
        }
        if (std::find(isPermuteElemTypeEquals.begin(), isPermuteElemTypeEquals.end(), false) !=
            isPermuteElemTypeEquals.end()) {
            return false;
        }

        auto output = eltwiseOp.getOutput();
        // When elementwise operation has two consumers with at least one IE.ShapeCast, skip such case.
        // Such branching leads to compilation failure.
        const auto isShapeCast = [](mlir::Operation* op) -> bool {
            return mlir::isa<IE::ShapeCastOp>(op);
        };
        const auto hasShapeCastConsumer = llvm::any_of(output.getUsers(), isShapeCast);
        if (!output.hasOneUse() && hasShapeCastConsumer) {
            return false;
        }

        return true;
    };

    auto alignIface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(eltwiseOp.getOperation());
    const auto channelAlignment = alignIface != nullptr ? alignIface.getInputChannelAlignment() : 1;
    const auto isChannelAligned = [&](ShapeRef shape) {
        return shape[Dims4D::Act::C] % channelAlignment == 0;
    };

    const auto patternCanAvoidShapeCast = [&](ShapeRef mappedShape) {
        if (auto groupConvOp = mlir::dyn_cast<IE::GroupConvolutionOp>(*eltwiseOp)) {
            if (getShape(groupConvOp.getFilter()).totalSize() != 1) {
                return false;
            }
        }
        const auto isParentOfThisEltwise = [&](mlir::Operation* op) -> bool {
            return !VPU::hasMultiBranches(op) && *(op->getUsers().begin()) == eltwiseOp;
        };
        if (!llvm::all_of(inputPermutes, isParentOfThisEltwise)) {
            return false;
        }
        if (eltwiseOp->hasOneUse() && mlir::isa<IE::ShapeCastOp>(*(eltwiseOp->getUsers().begin()))) {
            return false;
        }
        return isChannelAligned(mappedShape);
    };

    if (!patternCanBeConverted()) {
        return mlir::failure();
    }
    _log.nest().trace("Moving permute op post eltwise {0} at {1}", eltwiseOp->getName(), eltwiseOp->getLoc());
    auto permutesToMove = getPermutesToMove(inputPermutes);
    const auto neutralMemPerm =
            mlir::AffineMapAttr::get(mlir::AffineMap::getPermutationMap(SmallVector<unsigned>{0, 1, 2, 3}, ctx));
    // Get operation skipping QuantizeCast and ShapeCast ops
    auto getOwnerIgnoreCasts = [&](const mlir::OpOperand& opOperand) -> mlir::Operation* {
        auto ownerOp = opOperand.getOwner();
        while (ownerOp && mlir::isa<IE::QuantizeCastOp, IE::ShapeCastOp>(ownerOp) &&
               !ownerOp->getResult(0).getUsers().empty()) {
            ownerOp = *ownerOp->getResult(0).getUsers().begin();
        }
        return ownerOp;
    };

    auto mappedShape = getMappedShape(permuteInputLayouts[0], eltwiseInputLayout, permuteMemPermLayouts[0],
                                      getShape(inputPermutes[0]->getOperand(0)));
    auto canAvoidShapeCast = patternCanAvoidShapeCast(mappedShape);

    for (auto curPermute : permutesToMove) {
        _log.nest().trace("Processing permute {0} {1}", curPermute->getName(), curPermute->getLoc());
        auto permuteOutputType = curPermute->getResult(0).getType().template cast<vpux::NDTypeInterface>();
        const auto dstOrder = mlir::AffineMapAttr::get(eltwiseInputLayout.toAffineMap(ctx));
        rewriter.setInsertionPoint(curPermute);
        mlir::Value outputVal;
        if (permuteInputLayouts[0] != eltwiseInputLayout) {
            auto permuteCast = rewriter.template create<IE::PermuteCastOp>(
                    curPermute->getLoc(), curPermute->getOperand(0), dstOrder, neutralMemPerm);
            auto newPermuteCastOutputType = permuteCast.getOutput().getType().template cast<vpux::NDTypeInterface>();
            mappedShape = newPermuteCastOutputType.getShape().toValues();
            canAvoidShapeCast = canAvoidShapeCast && isChannelAligned(mappedShape);
            outputVal = permuteCast.getResult();
            if (!canAvoidShapeCast) {
                outputVal = rewriter.template create<IE::ShapeCastOp>(
                        curPermute->getLoc(), newPermuteCastOutputType.changeShape(eltwiseInputShape),
                        permuteCast.getOutput(), getIntArrayAttr(ctx, eltwiseInputShape.raw()));
            }
        } else {
            outputVal = rewriter.template create<IE::ShapeCastOp>(
                    curPermute->getLoc(), permuteOutputType.changeShape(eltwiseInputShape), curPermute->getOperand(0),
                    getIntArrayAttr(ctx, eltwiseInputShape.raw()));
        }
        curPermute->getResult(0).replaceUsesWithIf(outputVal, [&](mlir::OpOperand& opOperand) {
            return getOwnerIgnoreCasts(opOperand) == eltwiseOp;
        });
    }

    // Get the output value of the last QuantizeCast op
    // or the output value of the Eltwise op if no QuantizeCast op exists
    auto getInsertPoint = [&]() -> mlir::Value {
        auto output = eltwiseOp.getOutput();
        if (output.getUsers().empty()) {
            return output;
        }
        // When there are more than one consumer, return the output eltwise.
        if (!output.hasOneUse()) {
            return output;
        }
        if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*output.getUsers().begin())) {
            return quantizeCastOp.getOutput();
        }
        if (auto shapeCastOp = mlir::dyn_cast<IE::ShapeCastOp>(*output.getUsers().begin())) {
            // Same here, when IE.ShapeCast has several consumers, bail out.
            if (!shapeCastOp.getResult().hasOneUse()) {
                return shapeCastOp.getResult();
            }
            // In case IE.Add -> IE.ShapeCast -> IE.QuantizeCast, return QuantizeCast.
            if (auto quantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*shapeCastOp.getResult().getUsers().begin())) {
                return quantizeCastOp.getOutput();
            }

            return shapeCastOp.getResult();
        }
        return output;
    };
    auto outputValue = getInsertPoint();
    auto eltwiseOutputType = outputValue.getType().template cast<vpux::NDTypeInterface>();
    rewriter.setInsertionPointAfter(outputValue.getDefiningOp());
    auto newOutputShapeCastType = eltwiseOutputType.changeShape(mappedShape);
    // Get new output memPermuteOp or permuteQuantizeOp
    auto getOutputPermute = [&](const mlir::Operation* op,
                                const mlir::Value inputValue) -> mlir::FailureOr<mlir::Operation*> {
        if (auto memPermute = mlir::dyn_cast<IE::MemPermuteOp>(op)) {
            return rewriter
                    .template create<IE::MemPermuteOp>(eltwiseOp->getLoc(), inputValue, memPermute.getDstOrderAttr(),
                                                       memPermute.getMemPermAttr())
                    .getOperation();
        } else if (auto permuteQuantize = mlir::dyn_cast<IE::PermuteQuantizeOp>(op)) {
            auto dstElemType = permuteQuantize.getDstElemType();
            if (canAvoidShapeCast && (dstElemType.isF16() || dstElemType.isF32())) {
                // if no extra shape cast needed and output permuteQuantize is actually a pure memPermute,
                // create memPermute here so later it can be fused into ODU of the eltwise
                return rewriter
                        .template create<IE::MemPermuteOp>(eltwiseOp->getLoc(), inputValue,
                                                           permuteQuantize.getDstOrderAttr(),
                                                           permuteQuantize.getMemPermAttr())
                        .getOperation();
            } else {
                return rewriter
                        .template create<IE::PermuteQuantizeOp>(
                                eltwiseOp->getLoc(), inputValue, permuteQuantize.getDstOrderAttr(),
                                permuteQuantize.getMemPermAttr(), permuteQuantize.getDstElemTypeAttr(),
                                permuteQuantize.getPadsBeginAttr(), permuteQuantize.getPadsEndAttr())
                        .getOperation();
            }
        } else {
            VPUX_THROW("Unsupported operation, operation should be MemPermuteOp or PermuteQuantizeOp!");
        }
    };

    if (permuteInputLayouts[0] != eltwiseInputLayout) {
        if (canAvoidShapeCast) {
            // Set output shape of all ops (here actually only QuantizeCastOp) between EltwiseOp(included)
            // and output PermuteOp to mapped shape.
            // For example, for a case like "MemPermuteOp -> EltwiseOp -> QuantizeCastOp -> ...", if we want to
            // move the MemPermuteOp after QuantizeCastOp without adding ShapeCast after EltwiseOp, we need to
            // set the mapped shape to output of both EltwiseOp and QuantizeCastOp.
            auto currOutput = outputValue;
            while (true) {
                const auto currOutputType = currOutput.getType().template cast<vpux::NDTypeInterface>();
                const auto newOutputType = currOutputType.changeShape(mappedShape);
                currOutput.setType(newOutputType);
                auto parentOp = currOutput.getDefiningOp();
                VPUX_THROW_WHEN(parentOp == nullptr, "The connections were broken");
                if (parentOp == eltwiseOp) {
                    break;
                }
                currOutput = parentOp->getOperand(0);
            }
        }
        auto outputShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType, outputValue, getIntArrayAttr(ctx, mappedShape.raw()));
        const auto outputDstOrder = mlir::AffineMapAttr::get(permuteInputLayouts[0].toAffineMap(ctx));
        auto outputPermuteCast = rewriter.template create<IE::PermuteCastOp>(
                eltwiseOp->getLoc(), outputShapeCast.getResult(), outputDstOrder, neutralMemPerm);
        auto permuteOrFailure = getOutputPermute(inputPermutes[0], outputPermuteCast.getOutput());
        if (mlir::failed(permuteOrFailure)) {
            return mlir::failure();
        }
        auto outputPermute = permuteOrFailure.value();
        auto eltwiseOutputShape = eltwiseOutputType.getShape();
        auto outputPermuteOutputShape = newOutputShapeCastType.getShape();
        if (eltwiseOutputShape != outputPermuteOutputShape) {
            /*
            For the specific case

               Permute    Permute
                  |          |
              ShapeCast   ShapeCast
                  |          |
            (QuantizeCast) (QuantizeCast)
                    \        /
                     Eltwise
                        |
                     Eltwise
                        |

                to:

            PermuteCast  PermuteCast
                  |          |
             ShapeCast   ShapeCast
                  |          |
            (QuantizeCast) (QuantizeCast)
                   \        /
                     Eltwise
                        |
                 (QuantizeCast)
                        |
                    ShapeCast
                        |
                   PermuteCast
                        |
                     Permute
                        |
                    ShapeCast
             There should be an extra ShapeCast.
            */
            auto outputPermuteShapeCast = rewriter.template create<IE::ShapeCastOp>(
                    eltwiseOp->getLoc(), eltwiseOutputType, outputPermute->getResult(0),
                    getIntArrayAttr(ctx, eltwiseOutputShape.raw()));
            outputValue.replaceAllUsesExcept(outputPermuteShapeCast, outputShapeCast);
        } else {
            outputPermute->getResult(0).setType(eltwiseOutputType);
            outputValue.replaceAllUsesExcept(outputPermute->getResult(0), outputShapeCast);
        }
    } else {
        auto outputShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType.changeShape(permuteInputTypes[0].getShape()), outputValue,
                getIntArrayAttr(ctx, permuteInputTypes[0].getShape().raw()));
        auto permuteOrFailure = getOutputPermute(inputPermutes[0], outputShapeCast.getResult());
        if (mlir::failed(permuteOrFailure)) {
            return mlir::failure();
        }
        auto outputPermute = permuteOrFailure.value();
        auto permuteOutShape =
                inputPermutes[0]->getResult(0).getType().template cast<vpux::NDTypeInterface>().getShape();
        auto outPermuteType = eltwiseOutputType.changeShape(permuteOutShape);
        outputPermute->getResult(0).setType(outPermuteType);
        auto permuteOutShapeCast = rewriter.template create<IE::ShapeCastOp>(
                eltwiseOp->getLoc(), newOutputShapeCastType.changeShape(eltwiseOutputType.getShape()),
                outputPermute->getResult(0), getIntArrayAttr(ctx, eltwiseOutputType.getShape().raw()));
        outputValue.replaceAllUsesExcept(permuteOutShapeCast, outputShapeCast);
    }

    return mlir::success();
}

void MovePermutePostEltwisePass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto verifyGroupConv = [](mlir::Operation* op) {
        auto groupConvOp = mlir::cast<IE::GroupConvolutionOp>(op);
        if (!isEltwiseGroupConvolution(groupConvOp)) {
            return false;
        }
        mlir::SmallVector<Const::DeclareOp> constInputOps;
        // Skip the first operand, check only weights and biases.
        for (unsigned operandIdx = 1; operandIdx < op->getNumOperands(); operandIdx++) {
            const mlir::Value operand = op->getOperand(operandIdx);
            Const::DeclareOp declareOp = operand.getDefiningOp<Const::DeclareOp>();
            constInputOps.push_back(declareOp);
        }
        return llvm::all_of(constInputOps, isSplatConstant);
    };

    const auto verifyAvgPool = [](mlir::Operation* op) {
        auto avgPoolOp = mlir::cast<IE::AvgPoolOp>(op);
        return isEltwisePooling<IE::AvgPoolOp>(avgPoolOp);
    };

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<PermuteEltwiseRewriter<IE::AddOp>>(&ctx, nullptr, 2, _log);
    patterns.add<PermuteEltwiseRewriter<IE::MultiplyOp>>(&ctx, nullptr, 2, _log);
    patterns.add<PermuteEltwiseRewriter<IE::SubtractOp>>(&ctx, nullptr, 2, _log);
    patterns.add<PermuteEltwiseRewriter<IE::GroupConvolutionOp>>(&ctx, verifyGroupConv, 1, _log);
    patterns.add<PermuteEltwiseRewriter<IE::AvgPoolOp>>(&ctx, verifyAvgPool, 1, _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
        return;
    }
}

}  // namespace

//
// createMovePermutePostEltwisePass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMovePermutePostEltwisePass(Logger log) {
    return std::make_unique<MovePermutePostEltwisePass>(log);
}
