//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/permute_quantize_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/permute_utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Support/LLVM.h>

#include <optional>
#include <tuple>
#include <utility>

namespace vpux {
namespace IE {

bool doesAffineReshapeChangeRank(IE::AffineReshapeOp reshape) {
    auto inputType = mlir::cast<vpux::NDTypeInterface>(reshape.getInput().getType());
    auto outputType = mlir::cast<vpux::NDTypeInterface>(reshape.getOutput().getType());
    return inputType.getRank() != outputType.getRank();
}

// Check if operation is a valid compute op (Conv/GroupConv or Eltwise)
bool isValidComputeOp(mlir::Operation* op) {
    if (op == nullptr) {
        return false;
    }
    return mlir::isa<IE::ConvolutionOp, IE::GroupConvolutionOp>(op) || op->hasTrait<IE::EltwiseOp>();
}

SmallVector<int64_t> invertDimMappingWithAxesNotSplitOrMerged(ArrayRef<SmallVector<int64_t>> dimMapping,
                                                              ShapeRef affineInShape, ShapeRef affineOutShape) {
    SmallVector<int64_t> invertedDimMapping(affineOutShape.size(), 0);

    for (size_t inDim = 0; inDim < dimMapping.size(); inDim++) {
        auto dimsArr = dimMapping[inDim];
        for (size_t i = 0; i < dimsArr.size(); i++) {
            auto outDim = dimsArr[i];
            if (affineInShape[Dim(inDim)] == affineOutShape[Dim(outDim)]) {
                invertedDimMapping[outDim] = inDim;
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
    auto affineInShapeVec = to_small_vector(getShape(affineReshapeOp.getInput()));
    auto affineOutShapeVec = to_small_vector(getShape(affineReshapeOp.getOutput()));
    const auto dimIsOne = [](int64_t dim) {
        return dim == 1;
    };
    affineInShapeVec.erase(std::remove_if(affineInShapeVec.begin(), affineInShapeVec.end(), dimIsOne),
                           affineInShapeVec.end());
    affineOutShapeVec.erase(std::remove_if(affineOutShapeVec.begin(), affineOutShapeVec.end(), dimIsOne),
                            affineOutShapeVec.end());
    if (affineInShapeVec == affineOutShapeVec) {
        return newSoftmaxAxis;
    }

    if (softmaxOp.getPadSize().has_value() && newSoftmaxAxis != softmaxAxis) {
        log.trace("Softmax axis is changed");
        return std::nullopt;
    }

    return newSoftmaxAxis;
}

std::tuple<MoveTransposeAffineReshapeThroughAdd::InputsMode, IE::TransposeOp, IE::AffineReshapeOp>
MoveTransposeAffineReshapeThroughAdd::checkAddInputsMode(IE::AddOp origOp) const {
    const auto input1 = origOp.getInput1();
    const auto input2 = origOp.getInput2();
    if (getShape(input1) != getShape(input2)) {
        return {InputsMode::Unsupported, nullptr, nullptr};
    }

    auto getTransposeAndReshape =
            [](mlir::Value inputValue) -> std::optional<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        if (mlir::isa<mlir::BlockArgument>(inputValue)) {
            return std::nullopt;
        }

        auto reshapeOp = inputValue.getDefiningOp<IE::AffineReshapeOp>();
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
            [&](mlir::Value input1,
                mlir::Value input2) -> mlir::FailureOr<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        if (input1 == input2 &&
            static_cast<size_t>(std::distance(input1.getUsers().begin(), input1.getUsers().end())) == 2) {
            auto getOps = getTransposeAndReshape(input1);
            if (getOps.has_value()) {
                return getOps.value();
            }
        }

        return mlir::failure();
    };
    auto inputsHaveTheSameParent = checkInputsHaveTheSameParent(input1, input2);
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
            [&](mlir::Value input1,
                mlir::Value input2) -> mlir::FailureOr<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        auto getOps1 = getTransposeAndReshape(input1);
        if (!getOps1.has_value()) {
            return mlir::failure();
        }

        auto getOps2 = getTransposeAndReshape(input2);
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

    auto symmetricalInputs = checkSymmetricalInputsWithTheSameTransposeAndReshape(input1, input2);
    if (mlir::succeeded(symmetricalInputs)) {
        return {InputsMode::Symmetry, symmetricalInputs.value().first, symmetricalInputs.value().second};
    }

    // Check asymmetrical inputs
    const auto isSupportedNonAffineReshapeInput = [](mlir::Value inputValue) {
        if (mlir::isa<mlir::BlockArgument>(inputValue)) {
            return true;
        }

        auto inputOp = inputValue.getDefiningOp();
        return mlir::isa_and_present<IE::SelectOp, IE::ConvertOp, Const::DeclareOp>(inputOp);
    };
    auto checkAsymmetricalInputs =
            [&](mlir::Value input1,
                mlir::Value input2) -> mlir::FailureOr<std::pair<IE::TransposeOp, IE::AffineReshapeOp>> {
        auto getOps1 = getTransposeAndReshape(input1);
        auto getOps2 = getTransposeAndReshape(input2);
        if (getOps1.has_value() && !getOps2.has_value() && input1.hasOneUse() &&
            isSupportedNonAffineReshapeInput(input2)) {
            return getOps1.value();
        }

        if (!getOps1.has_value() && getOps2.has_value() && input2.hasOneUse() &&
            isSupportedNonAffineReshapeInput(input1)) {
            return getOps2.value();
        }

        return mlir::failure();
    };
    auto asymmetricalInputs = checkAsymmetricalInputs(input1, input2);
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

    // More checking for asymmetric case
    const auto input1 = origOp.getInput1();
    const auto input2 = origOp.getInput2();
    IE::AffineReshapeOp affineReshapeInputOp = nullptr;
    mlir::Operation* nonAffineReshapeInput = nullptr;
    if (mlir::isa_and_present<IE::AffineReshapeOp>(input1.getDefiningOp())) {
        affineReshapeInputOp = input1.getDefiningOp<IE::AffineReshapeOp>();
        nonAffineReshapeInput = input2.getDefiningOp();
    } else if (mlir::isa_and_present<IE::AffineReshapeOp>(input2.getDefiningOp())) {
        affineReshapeInputOp = input2.getDefiningOp<IE::AffineReshapeOp>();
        nonAffineReshapeInput = input1.getDefiningOp();
    } else {
        return false;
    }

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
    if (mlir::isa_and_present<Const::DeclareOp>(nonAffineReshapeInput)) {
        return true;
    }

    /*
        For the Asymmetry case like below:
              NCEOp
                |
            Transpose     Operand
                |            |
          AffineReshape   Convert
                \          /
                    Add
                     |
                  [Concat]
                     |
                  Softmax
                     |

            will be converted into:

              NCEOp     Operand
                |          |
                |       Convert
                |          |
                |    AffineReshape
                |          |
                |      Transpose(will convert to PermuteCast)
                 \       /
                    Add
                     |
                  [Concat]
                     |
                  Softmax
                     |
                AffineReshape
                     |
                 Transpose
                     |

        Another input's transpose can be replaced by PermuteCast
        After the conversion, the Convolution, Add, SoftMax ir reorders will be changed, all NCEOps will in parallel
        with SoftMax, this will benefit SDPA case.
    */

    auto checkAsymmetricPatternWithSoftmax = [&]() -> bool {
        auto childOp = *origOp.getOutput().user_begin();
        if (auto maybeConcatOp = mlir::dyn_cast_if_present<IE::ConcatOp>(childOp)) {
            if (!maybeConcatOp->hasOneUse()) {
                return false;
            }
            childOp = *maybeConcatOp->getUsers().begin();
        }

        auto softmaxOp = mlir::dyn_cast<IE::SoftMaxOp>(childOp);
        if (softmaxOp == nullptr) {
            return false;
        }

        mlir::Value nonAffineReshapeInputValue = affineReshapeInputOp == input1.getDefiningOp() ? input2 : input1;

        auto checkOnlyTwoNonUnitDims = [](mlir::Value input) -> bool {
            auto inputShape = getShape(input);
            return std::count_if(inputShape.begin(), inputShape.end(), [](int64_t dim) {
                       return dim != 1;
                   }) == 2;
        };

        return checkOnlyTwoNonUnitDims(nonAffineReshapeInputValue);
    };

    if (checkAsymmetricPatternWithSoftmax()) {
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

    const auto input1 = origOp.getInput1();
    const auto input2 = origOp.getInput2();
    if (getShape(input1) != getShape(input2)) {
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
    const auto addInput = getShape(input1);
    if (reshapeInput.size() != addInput.size()) {
        return mlir::failure();
    }

    auto inputType = mlir::cast<vpux::NDTypeInterface>(transposeOp.getInput().getType());
    const auto inputShape = inputType.getShape();
    const auto alignment = VPU::NCEInvariant::getAlignment(inputType.getElementType());
    if (inputShape[Dims4D::Act::C] % alignment || inputShape[Dims4D::Act::N] > 1) {
        return mlir::failure();
    }

    // A per-channel scale tensor encodes one scale value per output channel. Moving a
    // Transpose/AffineReshape through the op when it modifies the channel dimension or
    // changes the memory layout would invalidate the per-channel scales.
    if (origOp.getScale() != nullptr) {
        if (inputShape[Dims4D::Act::C] != addInput[Dims4D::Act::C]) {
            _log.trace("Skipping '{0}': scale table present and channel dimension changes ({1} -> {2})",
                       origOp->getLoc(), addInput[Dims4D::Act::C], inputShape[Dims4D::Act::C]);
            return mlir::failure();
        }
        const auto perm = DimsOrder::fromAffineMap(transposeOp.getOrderValueAttr().getValue());
        if (perm.dimAt(Dims4D::Act::C.ind()) != Dims4D::Act::C) {
            _log.trace("Skipping '{0}': scale table present and transpose permutes the channel axis", origOp->getLoc());
            return mlir::failure();
        }
    }

    if (!isBeneficialConversion(origOp, inputsMode)) {
        return mlir::failure();
    }

    auto orderValueAttr = mlir::AffineMapAttr::get(
            vpux::getPermutationFromOrders(DimsOrder::fromAffineMap(transposeOp.getOrderValueAttr().getValue()),
                                           DimsOrder::NCHW, rewriter.getContext()));
    auto getInputValue = [&](mlir::Value inputValue) -> mlir::Value {
        if (inputsMode == InputsMode::Symmetry) {
            auto reshapeOp = inputValue.getDefiningOp<IE::AffineReshapeOp>();
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
        bool isBlockArg = mlir::isa<mlir::BlockArgument>(inputValue);
        if (!isBlockArg && mlir::isa<IE::AffineReshapeOp>(inputValue.getDefiningOp())) {
            auto reshapeOp = inputValue.getDefiningOp<IE::AffineReshapeOp>();
            auto transposeOp = reshapeOp.getInput().getDefiningOp<IE::TransposeOp>();
            VPUX_THROW_WHEN(transposeOp == nullptr, "Can't find TransposeOp");
            return transposeOp.getInput();
        }

        // Input is BlockArgument or not AffineReshape, build Reshape - Transpose on this as the new Add's input.
        auto constReshape = rewriter.createOrFold<IE::ReshapeOp>(takeOpLoc(origOp, "reshape_in"), inputValue,
                                                                 getIntArrayAttr(rewriter.getContext(), reshapeInput));
        return rewriter
                .create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_in"), constReshape, nullptr, orderValueAttr)
                .getResult();
    };

    auto newInput1 = getInputValue(input1);
    auto newInput2 = getInputValue(input2);
    auto origOutputType = mlir::cast<vpux::NDTypeInterface>(origOp.getOutput().getType());
    auto outElemType = origOutputType.getElementType();
    auto newAddOutType = mlir::cast<vpux::NDTypeInterface>(mlir::RankedTensorType::get(inputShape, outElemType));
    newAddOutType = newAddOutType.changeDimsOrder(origOutputType.getDimsOrder());
    auto outputVal =
            rewriter.create<IE::AddOp>(takeOpLoc(origOp, "as_add"), newAddOutType, newInput1, newInput2,
                                       origOp.getScale(), origOp.getAutoBroadcastAttr(), origOp.getPostOpAttr(),
                                       origOp.getClampAttr(), origOp.getStaticScaleAttr(),
                                       origOp.getOutputPaddingAttr(), origOp.getInputPaddingAttr())
                    .getOutput();

    auto postQuantizeCastOp = mlir::dyn_cast<IE::QuantizeCastOp>(*origOp.getOutput().user_begin());
    if (postQuantizeCastOp != nullptr && origOp->hasOneUse()) {
        outputVal = rewriter.create<IE::QuantizeCastOp>(
                                    takeOpLoc(origOp, "qcast"), outputVal,
                                    mlir::cast<vpux::NDTypeInterface>(postQuantizeCastOp.getOutput().getType())
                                            .getElementType())
                            .getOutput();
    }

    auto newTransposeOp = rewriter.create<IE::TransposeOp>(takeOpLoc(origOp, "transpose_out"), outputVal,
                                                           transposeOp.getOrder(), transposeOp.getOrderValueAttr());
    auto newReshapeOp = rewriter.create<IE::AffineReshapeOp>(
            takeOpLoc(origOp, "reshape_out"), newTransposeOp.getOutput(), affineReshapeOp.getDimMappingAttr(),
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

// Find the new Softmax axis after swapping ShapeCast and Softmax operations.
// The algorithm works on memory shape (permuted shape) to find the matching axis,
// then converts it back to logical shape axis for returning.
//
// To ensure the swap is valid, the new axis must satisfy:
// 1. The dimension size at the new axis equals the dimension size at the original axis
// 2. The product of dimensions to the left of the new axis equals the product to the left of the original axis
// 3. The product of dimensions to the right of the new axis equals the product to the right of the original axis
//
// Example 1 - Valid swap (working on memory shape):
//
//   Input logical shape:  [1, 16, 16, 1] with #NWCH -> memShape: [1, 1, 16, 16]
//   Output logical shape: [1, 1, 16, 16] with #NWCH -> memShape: [1, 16, 1, 16]
//   Original axis in logical shape: 2 -> in memShape: axis 3 (dim=16)
//
//   For original memAxis 3 (dim=16) in output memShape [1, 16, 1, 16]:
//     - Left product:  1 * 16 * 1 = 16
//     - Right product: 1
//
//   Searching in input memShape [1, 1, 16, 16] for matching axis:
//     - At memAxis 3: dim=16, left product = 1*1*16 = 16, right product = 1
//   memAxis 3 corresponds to logical axis 2 (d2 at position 3 in memory order)
//   After swap, the new SoftMax axis will be 2 in the input logical shape.
//
// Example 2 - Invalid swap (cannot find matching axis):
//
//   Input logical shape:  [1, 17, 16, 1] with #NWCH -> memShape: [1, 1, 17, 16]
//   Output logical shape: [1, 1, 17, 16] with #NWCH -> memShape: [1, 16, 1, 17]
//   Original axis in logical shape: 2 -> in memShape: axis 3 (dim=17)
//
//   For original memAxis 3 (dim=17) in output memShape [1, 16, 1, 17]:
//     - Left product:  1 * 16 * 1 = 16
//     - Right product: 1
//
//   Searching in input memShape [1, 1, 17, 16] for matching axis:
//     - At memAxis 2: dim=17, left product = 1*1 = 1 (need 16)
//     - At memAxis 3: dim=16, dimension mismatch (need 17)
//   No matching axis found, so this swap is invalid.
//
std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithShapeCast(int64_t softmaxLogicalAxis,
                                                                   IE::ShapeCastOp shapeCastOp, const Logger& log) {
    if (shapeCastOp == nullptr || !shapeCastOp->hasOneUse()) {
        log.trace("ShapeCastOp not found or has multiple uses");
        return std::nullopt;
    }

    const auto inMemShape = getMemShape(shapeCastOp.getInput());
    const auto outMemShape = getMemShape(shapeCastOp.getOutput());
    const auto inputType = mlir::cast<vpux::NDTypeInterface>(shapeCastOp.getInput().getType());
    const auto outputType = mlir::cast<vpux::NDTypeInterface>(shapeCastOp.getOutput().getType());
    const auto inDimsOrder = inputType.getDimsOrder();
    const auto outDimsOrder = outputType.getDimsOrder();
    auto softmaxMemAxis = outDimsOrder.toMemDim(Dim(softmaxLogicalAxis)).ind();

    // Calculate the product of dimensions before and after the softmax axis in output memory shape
    int64_t leftProduct = 1;
    for (size_t idx = 0; idx < outMemShape.size(); idx++) {
        if (idx < static_cast<size_t>(softmaxMemAxis)) {
            leftProduct *= outMemShape[MemDim(idx)];
        }
    }

    // Find the corresponding axis in input memory shape by matching dimension size and position
    int64_t currentLeftProduct = 1;
    for (size_t inMemIdx = 0; inMemIdx < inMemShape.size(); inMemIdx++) {
        const int64_t currentDimSize = inMemShape[MemDim(inMemIdx)];
        if (currentDimSize == outMemShape[MemDim(softmaxMemAxis)]) {
            if (currentLeftProduct == leftProduct) {
                // Note: padSize is safe even when logical axis changes, because:
                // - padSize operates on the dimension in memory layout
                // - Since we matched the memory axis with same dimension size and products,
                //   the same elements in memory are affected by padSize
                auto newLogicalAxis = inDimsOrder.toDim(MemDim(inMemIdx)).ind();
                return newLogicalAxis;
            }
        }
        currentLeftProduct *= currentDimSize;
    }

    log.trace("Not found valid softmax axis after swapping with ShapeCast");
    return std::nullopt;
}

std::optional<int64_t> getNewSoftmaxAxisAfterSwappingWithShapeCast(IE::SoftMaxOp softmaxOp, IE::ShapeCastOp shapeCastOp,
                                                                   const Logger& log) {
    if (shapeCastOp == nullptr) {
        log.trace("ShapeCastOp is null");
        return std::nullopt;
    }
    const auto softmaxLogicalAxis = getPositiveAxisInd(softmaxOp.getAxisIndAttr(),
                                                       checked_cast<int64_t>(getShape(shapeCastOp.getOutput()).size()));
    return getNewSoftmaxAxisAfterSwappingWithShapeCast(softmaxLogicalAxis, shapeCastOp, log);
}

}  // namespace IE
}  // namespace vpux
