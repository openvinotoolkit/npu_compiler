//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include <mlir/Support/LLVM.h>
#include <optional>
#include <tuple>
#include <utility>
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {
namespace IE {

bool doesAffineReshapeChangeRank(IE::AffineReshapeOp reshape) {
    auto inputType = reshape.getInput().getType().cast<NDTypeInterface>();
    auto outputType = reshape.getOutput().getType().cast<NDTypeInterface>();
    return inputType.getRank() != outputType.getRank();
}

SmallVector<int64_t> invertDimMappingWithAxesNotSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping,
                                                              ShapeRef affineInShape, ShapeRef affineOutShape) {
    SmallVector<int64_t> invertedDimMapping(affineOutShape.size(), 0);

    for (size_t inDim = 0; inDim < dimMapping.size(); inDim++) {
        auto dimsArr = dimMapping[inDim];
        for (size_t i = 0; i < dimsArr.size(); i++) {
            auto outDim = dimsArr[i];
            if (affineInShape[Dim(inDim)] == affineOutShape[Dim(outDim)]) {
                invertedDimMapping[dimsArr[i]] = inDim;
                break;
            }
        }
    }

    return invertedDimMapping;
}

bool areModifiedAxesSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping, ShapeRef affineInShape,
                                  ShapeRef affineOutShape, const mlir::DenseSet<int64_t>& modifiedAxes, bool swapOrder,
                                  Logger log) {
    for (size_t inIdx = 0; inIdx < dimMapping.size(); inIdx++) {
        auto mappedDim = dimMapping[inIdx];

        for (size_t mapId = 0; mapId < mappedDim.size(); mapId++) {
            size_t outIdx = mappedDim[mapId];
            if (swapOrder) { /*Op->AffineReshape*/
                if (modifiedAxes.contains(inIdx)) {
                    if (affineOutShape[Dim(outIdx)] != 1 && affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)]) {
                        log.trace("Modified axis '{0}' was split or merged from several axes.", inIdx);
                        return true;
                    }
                }
            } else { /*AffineReshape->Op*/
                if (modifiedAxes.contains(outIdx)) {
                    if (affineInShape[Dim(inIdx)] != 1 && affineInShape[Dim(inIdx)] != affineOutShape[Dim(outIdx)]) {
                        log.trace("Modified axis '{0}' was split or merged from several axes.", outIdx);
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithAffineReshape(IE::SoftMaxOp softmaxOp,
                                                                       IE::AffineReshapeOp affineReshapeOp,
                                                                       const Logger& log) {
    if (affineReshapeOp == nullptr || !affineReshapeOp->hasOneUse()) {
        log.trace("AffineReshapeOp not found or has multiple uses");
        return std::nullopt;
    }

    if (doesAffineReshapeChangeRank(affineReshapeOp)) {
        log.trace("AffineReshapeOp should not change tensor rank");
        return std::nullopt;
    }

    const auto affineInShape = getShape(affineReshapeOp.getInput());
    const auto affineOutShape = getShape(affineReshapeOp.getOutput());
    const auto dimMapping = parseIntArrayOfArrayAttr<int64_t>(affineReshapeOp.getDimMapping());

    const auto softmaxAxis =
            getPositiveAxisInd(softmaxOp.getAxisIndAttr(), checked_cast<int64_t>(affineOutShape.size()));
    const mlir::DenseSet<int64_t> modifiedAxes{softmaxAxis};
    if (IE::areModifiedAxesSplitOrMerged(dimMapping, affineInShape, affineOutShape, modifiedAxes, false, log)) {
        return std::nullopt;
    }

    const auto invertedDimMapping =
            IE::invertDimMappingWithAxesNotSplitOrMerged(dimMapping, affineInShape, affineOutShape);
    const auto newSoftmaxAxis = invertedDimMapping[softmaxAxis];
    if (softmaxOp.getPadSize().has_value() && newSoftmaxAxis != softmaxAxis) {
        log.trace("Softmax axis is changed");
        return std::nullopt;
    }

    return newSoftmaxAxis;
}

std::tuple<MoveTransposeAffineReshapeThroughAdd::InputsMode, IE::TransposeOp, IE::AffineReshapeOp>
MoveTransposeAffineReshapeThroughAdd::checkAddInputsMode(IE::AddOp origOp) const {
    auto input1Op = origOp.getInput1().getDefiningOp();
    auto input2Op = origOp.getInput2().getDefiningOp();
    if (input1Op == nullptr || input2Op == nullptr || getShape(origOp.getInput1()) != getShape(origOp.getInput2())) {
        return {InputsMode::Unsupported, nullptr, nullptr};
    }

    auto getTransposeAndReshape =
            [](mlir::Operation* inputOp) -> std::optional<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        auto reshapeOp = mlir::dyn_cast_or_null<IE::AffineReshapeOp>(inputOp);
        if (reshapeOp == nullptr) {
            return std::nullopt;
        }

        auto transposeOp = reshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
        if (transposeOp == nullptr || !transposeOp->hasOneUse()) {
            return std::nullopt;
        }

        return std::make_pair(transposeOp, reshapeOp);
    };

    // Two inputs of add are from the same affineReshape
    //      Transpose
    //          |
    //    AffineReshapeOp
    //         | |
    //         Add
    //          |
    auto checkInputsHaveTheSameParent =
            [&](mlir::Operation* input1Op,
                mlir::Operation* input2Op) -> mlir::FailureOr<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        if (input1Op == input2Op &&
            static_cast<size_t>(std::distance(input1Op->getUsers().begin(), input1Op->getUsers().end())) == 2) {
            auto getOps = getTransposeAndReshape(input1Op);
            if (getOps.has_value()) {
                return getOps.value();
            }
        }

        return mlir::failure();
    };
    auto inputsHaveTheSameParent = checkInputsHaveTheSameParent(input1Op, input2Op);
    if (mlir::succeeded(inputsHaveTheSameParent)) {
        return {InputsMode::Symmetry, inputsHaveTheSameParent.value().first, inputsHaveTheSameParent.value().second};
    }

    // Each input of Add has Transpose and AffineReshape, and these Transpose / AffineReshape ops are the same
    //      Transpose       Transpose
    //          |               |
    //    AffineReshape    AffineReshape
    //          \               /
    //                 Add
    //                  |
    auto checkSymmetricalInputsWithTheSameTransposeAndReshape =
            [&](mlir::Operation* input1Op,
                mlir::Operation* input2Op) -> mlir::FailureOr<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        auto getOps1 = getTransposeAndReshape(input1Op);
        if (!getOps1.has_value()) {
            return mlir::failure();
        }

        auto getOps2 = getTransposeAndReshape(input2Op);
        if (!getOps2.has_value()) {
            return mlir::failure();
        }

        auto reshape1 = getOps1.value().second;
        auto reshape2 = getOps2.value().second;
        if (!reshape1->hasOneUse() || !reshape2->hasOneUse()) {
            return mlir::failure();
        }

        auto areTheSameReshapeOps = reshape1.getInput().getType() == reshape2.getInput().getType() &&
                                    reshape1.getOutput().getType() == reshape2.getOutput().getType();
        if (!areTheSameReshapeOps) {
            return mlir::failure();
        }

        auto tranpose1 = getOps1.value().first;
        auto tranpose2 = getOps2.value().first;
        if (!tranpose1.getOrderValue().has_value() || !tranpose2.getOrderValue().has_value()) {
            return mlir::failure();
        }

        auto areTheSameTransposeOps = tranpose1.getInput().getType() == tranpose2.getInput().getType() &&
                                      tranpose1.getOrderValue().value() == tranpose2.getOrderValue().value();
        if (!areTheSameTransposeOps) {
            return mlir::failure();
        }

        return getOps1.value();
    };

    auto symmetricalInputs = checkSymmetricalInputsWithTheSameTransposeAndReshape(input1Op, input2Op);
    if (mlir::succeeded(symmetricalInputs)) {
        return {InputsMode::Symmetry, symmetricalInputs.value().first, symmetricalInputs.value().second};
    }

    // Check asymmetrical inputs
    const auto isSupportedNonAffineReshapeInput = [](mlir::Operation* inputOp) {
        return mlir::isa_and_nonnull<IE::SelectOp, IE::ConvertOp, Const::DeclareOp>(inputOp);
    };
    auto checkAsymmetricalInputs =
            [&](mlir::Operation* input1Op,
                mlir::Operation* input2Op) -> mlir::FailureOr<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        auto getOps1 = getTransposeAndReshape(input1Op);
        auto getOps2 = getTransposeAndReshape(input2Op);
        if (getOps1.has_value() && !getOps2.has_value() && input1Op->hasOneUse() &&
            isSupportedNonAffineReshapeInput(input2Op)) {
            return getOps1.value();
        }

        if (!getOps1.has_value() && getOps2.has_value() && input2Op->hasOneUse() &&
            isSupportedNonAffineReshapeInput(input1Op)) {
            return getOps2.value();
        }

        return mlir::failure();
    };
    auto asymmetricalInputs = checkAsymmetricalInputs(input1Op, input2Op);
    if (mlir::succeeded(asymmetricalInputs)) {
        return {InputsMode::Asymmetry, asymmetricalInputs.value().first, asymmetricalInputs.value().second};
    }

    // Unsupported case
    return {InputsMode::Unsupported, nullptr, nullptr};
}

bool MoveTransposeAffineReshapeThroughAdd::isBeneficialConversion(IE::AddOp origOp, InputsMode mode) const {
    if (!origOp->hasOneUse()) {
        return false;
    }

    /*
        For AddOp with symmetrical Transpose & AffineReshape inputs cases, no need to check output given no additional
       transpose is introduced in.

        Transpose                     Transpose       Transpose
            |                             |               |
        AffineReshape       and     AffineReshape    AffineReshape
           | |                            \               /
           Add                                   Add
            |                                     |

        will be converted into:

         Operand                        Operand     Operand
           | |                              \           /
           Add                                   Add
            |               and                   |
        Transpose                             Transpose
            |                                     |
        AffineReshape                       AffineReshape
            |                                     |
    */
    if (mode == InputsMode::Symmetry) {
        return true;
    }

    auto affineReshapeInputOp = origOp.getInput1().getDefiningOp<IE::AffineReshapeOp>();
    auto nonAffineReshapeInput = origOp.getInput2().getDefiningOp();
    if (affineReshapeInputOp == nullptr) {
        affineReshapeInputOp = origOp.getInput2().getDefiningOp<IE::AffineReshapeOp>();
        nonAffineReshapeInput = origOp.getInput1().getDefiningOp();
    }
    VPUX_THROW_WHEN(affineReshapeInputOp == nullptr, "Can't find AffineReshapeOp input");

    /*
        If another input is Constant, new Transpose and AffineShape can be fused into constant.
        No extra Transpose operation is introduced in, no need to check output pattern.

        Transpose   Constant
            |         |
        AffineReshape |
            \         /
                Add
                 |

        will be converted into:

        Operand   Constant[Reshape, Reorder]
            |         |
            |         |
            \         /
                Add
                 |
             Transpose
                 |
            AffineReshape
                 |
    */
    if (mlir::isa<Const::DeclareOp>(nonAffineReshapeInput)) {
        return true;
    }

    /*
        For other generic pattern like below, we need to check if number of Transpose operations is not increased
        after conversion.

        Transpose   Operand
            |           |
        AffineReshape   |
            \         /
                Add
                 |
             [Softmax]
                 |
            AffineReshap
                 |
             Transpose
                 |

        will be converted into:

                    Operand
            |         |
            |      Reshape
            |         |
            |     Transpose
            \         /
                Add
                 |
             Transpose
                 |
            AffineReshap
                 |
             [Softmax]
                 |
            AffineReshap
                 |
             Transpose
                 |

        There's no extra Transpose operation after conversion when:
        1. Softmax (if it exists) is eligible to swap with parenet AffineReshape
        2. Output adjacent AffineReshape operations can be folded.
    */
    // Find child AffineReshape and Transpose
    auto childOp = *origOp.getOutput().user_begin();
    auto childAffineReshape = mlir::dyn_cast<IE::AffineReshapeOp>(childOp);
    if (childAffineReshape == nullptr) {
        if (!mlir::isa<IE::QuantizeCastOp, IE::SoftMaxOp>(childOp) || !childOp->hasOneUse()) {
            return false;
        }
        childOp = *childOp->getUsers().begin();
        childAffineReshape = mlir::dyn_cast<IE::AffineReshapeOp>(childOp);
    }
    if (childAffineReshape == nullptr || !childAffineReshape->hasOneUse()) {
        return false;
    }

    auto childTranspose = mlir::dyn_cast<IE::TransposeOp>(*childAffineReshape->getUsers().begin());
    if (childTranspose == nullptr) {
        return false;
    }

    // Ensure child Softmax is eligible to swap with parenet AffineReshape
    auto maybeSoftmaxOp = mlir::dyn_cast<IE::SoftMaxOp>(*origOp.getOutput().user_begin());
    if (maybeSoftmaxOp != nullptr && IE::getNewSoftmaxAxisAfterSwappingWithAffineReshape(
                                             maybeSoftmaxOp, affineReshapeInputOp, _log) == std::nullopt) {
        return false;
    }

    // Ensure no extra Tranpose will be introduced in:
    // if two adjacent AffineShape layers can be folded, then two Transpose layers can be fused into one or even
    // cancel each other.
    auto affineReshapeOpsCanBeFolded =
            affineReshapeInputOp.getInput().getType() == childAffineReshape.getOutput().getType();
    return affineReshapeOpsCanBeFolded;
};

mlir::LogicalResult MoveTransposeAffineReshapeThroughAdd::matchAndRewrite(IE::AddOp origOp,
                                                                          mlir::PatternRewriter& rewriter) const {
    _log.trace("Got '{0}' at '{1}'", origOp->getName(), origOp->getLoc());

    auto input1Op = origOp.getInput1().getDefiningOp();
    auto input2Op = origOp.getInput2().getDefiningOp();

    if (input1Op == nullptr || input2Op == nullptr || getShape(origOp.getInput1()) != getShape(origOp.getInput2())) {
        return mlir::failure();
    }

    InputsMode inputsMode = InputsMode::Unsupported;
    IE::TransposeOp transposeOp;
    IE::AffineReshapeOp affineReshapeOp;
    std::tie(inputsMode, transposeOp, affineReshapeOp) = checkAddInputsMode(origOp);
    if (inputsMode == InputsMode::Unsupported) {
        return mlir::failure();
    }

    const auto reshapeInput = getShape(affineReshapeOp.getInput());
    const auto addInput = getShape(origOp.getInput1());
    if (reshapeInput.size() != addInput.size()) {
        return mlir::failure();
    }

    auto inputType = transposeOp.getInput().getType().cast<vpux::NDTypeInterface>();
    const auto inputShape = inputType.getShape();
    const auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    if (inputShape[Dims4D::Act::C] % alignment || inputShape[Dims4D::Act::N] > 1) {
        return mlir::failure();
    }

    if (!isBeneficialConversion(origOp, inputsMode)) {
        return mlir::failure();
    }

    auto orderValueAttr = mlir::AffineMapAttr::get(
            vpux::getPermutationFromOrders(DimsOrder::fromAffineMap(transposeOp.getOrderValueAttr().getValue()),
                                           DimsOrder::NCHW, rewriter.getContext()));
    auto getInputValue = [&](mlir::Operation* op) -> mlir::Value {
        if (inputsMode == InputsMode::Symmetry) {
            auto reshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(op);
            VPUX_THROW_WHEN(reshapeOp == nullptr, "Can't find AffineReshapeOp");
            auto transposeOp = reshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
            VPUX_THROW_WHEN(transposeOp == nullptr, "Can't find TransposeOp");

            return transposeOp.getInput();
        }

        // For asymmetrical inputs:
        // If there are no Transpose - AffineReshape on the original input, then build Reshape - Transpose on this as
        // the new Add's input.
        // If there are Transpose - AffineReshape already, return the original TranposeOp input directly as the new
        // Add's input.
        auto reshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(op);
        if (reshapeOp == nullptr) {
            auto constReshape =
                    rewriter.createOrFold<IE::ReshapeOp>(origOp.getLoc(), op->getResult(0), nullptr, false,
                                                         getIntArrayAttr(rewriter.getContext(), reshapeInput));
            return rewriter.create<IE::TransposeOp>(origOp.getLoc(), constReshape, nullptr, orderValueAttr).getResult();
        } else {
            auto transposeOp = reshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
            VPUX_THROW_WHEN(transposeOp == nullptr, "Can't find TransposeOp");
            return transposeOp.getInput();
        }
    };

    auto input1 = getInputValue(input1Op);
    auto input2 = getInputValue(input2Op);
    auto origOutputType = origOp.getOutput().getType().cast<vpux::NDTypeInterface>();
    auto outElemType = origOutputType.getElementType();
    auto newAddOutType = mlir::RankedTensorType::get(inputShape, outElemType).cast<NDTypeInterface>();
    newAddOutType = newAddOutType.changeDimsOrder(origOutputType.getDimsOrder());
    auto outputVal =
            rewriter.create<IE::AddOp>(origOp.getLoc(), newAddOutType, input1, input2, origOp.getAutoBroadcastAttr(),
                                       origOp.getPostOpAttr(), origOp.getClampAttr(), origOp.getOutputChannelsAttr(),
                                       origOp.getInputChannelsAttr())
                    .getOutput();

    auto postQuantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*origOp.getOutput().user_begin());
    if (postQuantizeCastOp != nullptr && origOp->hasOneUse()) {
        outputVal =
                rewriter.create<IE::QuantizeCastOp>(
                                postQuantizeCastOp.getLoc(), outputVal,
                                postQuantizeCastOp.getOutput().getType().cast<vpux::NDTypeInterface>().getElementType())
                        .getOutput();
    }

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(transposeOp.getLoc(), outputVal, transposeOp.getOrder(),
                                                           transposeOp.getOrderValueAttr());
    auto newReshapeOp = rewriter.create<IE::AffineReshapeOp>(affineReshapeOp.getLoc(), newTransposeOp.getOutput(),
                                                             affineReshapeOp.getDimMappingAttr(),
                                                             affineReshapeOp.getShapeValueAttr());

    if (postQuantizeCastOp == nullptr) {
        origOp.replaceAllUsesWith(newReshapeOp.getOutput());
    } else {
        postQuantizeCastOp.replaceAllUsesWith(newReshapeOp.getOutput());
        rewriter.eraseOp(postQuantizeCastOp);
    }

    if (origOp) {
        rewriter.eraseOp(origOp);
    }
    if (affineReshapeOp) {
        rewriter.eraseOp(affineReshapeOp);
    }
    if (transposeOp) {
        rewriter.eraseOp(transposeOp);
    }

    _log.trace("[{0}] Replaced with 'IE::AffineReshapeOp'", getDebugName());
    return mlir::success();
}

}  // namespace IE
}  // namespace vpux
