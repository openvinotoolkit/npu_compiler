//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/rewriters/propagate_transpose_affine_reshape_common.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOSCALESHIFT
#define GEN_PASS_DEF_CONVERTTOSCALESHIFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// These handle cases where weights have spatial dimensions that need to be moved to channel position for ScaleShift
// H_EXPAND_PERMUTATION: For weights with shape [1, 1, H, 1] where H > 1
// Transforms [N=1, C=1, H=spatial, W=1] -> [N=1, C=spatial, H=1, W=1] via permutation (0,2,1,3)
// Example: [1, 1, 64, 1] -> [1, 64, 1, 1] to make it compatible with ScaleShift channel broadcasting
constexpr std::array<unsigned, 4> H_EXPAND_PERMUTATION = {0, 2, 1, 3};  // [N, H, C, W]
// W_EXPAND_PERMUTATION: For weights with shape [1, 1, 1, W] where W > 1
// Transforms [N=1, C=1, H=1, W=spatial] -> [N=1, C=spatial, H=1, W=1] via permutation (0,3,1,2)
// Example: [1, 1, 1, 128] -> [1, 128, 1, 1] to make it compatible with ScaleShift channel broadcasting
constexpr std::array<unsigned, 4> W_EXPAND_PERMUTATION = {0, 3, 1, 2};  // [N, W, C, H]
// H_EXPAND_INVERSE: Restores original layout after ScaleShift operation for H-expanded case
// Since H_EXPAND_PERMUTATION is self-inverse (swaps positions 1 and 2), inverse is same as forward
constexpr std::array<unsigned, 4> H_EXPAND_INVERSE = {0, 2, 1, 3};  // Same as forward for H
// W_EXPAND_INVERSE: Restores original layout after ScaleShift operation for W-expanded case
// Reverses W_EXPAND_PERMUTATION: [N=1, C=result, H=1, W=1] -> [N=1, C=1, H=1, W=result]
constexpr std::array<unsigned, 4> W_EXPAND_INVERSE = {0, 2, 3, 1};  // Inverse for W

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

struct TransposeInfo {
    SmallVector<unsigned> permutation;
    SmallVector<unsigned> inverse;
    bool isHExpanded;

    TransposeInfo(bool hExpanded): isHExpanded(hExpanded) {
        if (hExpanded) {
            permutation = {H_EXPAND_PERMUTATION.begin(), H_EXPAND_PERMUTATION.end()};
            inverse = {H_EXPAND_INVERSE.begin(), H_EXPAND_INVERSE.end()};
        } else {
            permutation = {W_EXPAND_PERMUTATION.begin(), W_EXPAND_PERMUTATION.end()};
            inverse = {W_EXPAND_INVERSE.begin(), W_EXPAND_INVERSE.end()};
        }
    }
};

// Helper function to check if weights need spatial transpose
std::optional<TransposeInfo> needsSpatialTranspose(ShapeRef weightsShape) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (weightsShape.size() != 4 || weightsShape[N] != 1 || weightsShape[C] != 1) {
        return std::nullopt;
    }

    const bool hExpanded = (weightsShape[H] > 1 && weightsShape[W] == 1);
    const bool wExpanded = (weightsShape[H] == 1 && weightsShape[W] > 1);

    if (!hExpanded && !wExpanded) {
        return std::nullopt;
    }

    return TransposeInfo(hExpanded);
}

// Helper function to compute transposed shape
SmallVector<int64_t> computeTransposedShape(ShapeRef originalShape, const TransposeInfo& info) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (info.isHExpanded) {
        return {originalShape[N], originalShape[H], originalShape[C], originalShape[W]};
    } else {
        return {originalShape[N], originalShape[W], originalShape[C], originalShape[H]};
    }
}

// Helper function to validate if transpose operations would be feasible without creating them
mlir::LogicalResult validateTransposeOperations(ShapeRef weightsShape, ShapeRef activationShape,
                                                const TransposeInfo& transposeInfo) {
    // Compute what the transposed shapes would be
    auto transposedWeightsShape = computeTransposedShape(weightsShape, transposeInfo);
    auto transposedActivationShape = computeTransposedShape(activationShape, transposeInfo);

    // Check basic constraints that verifyAndBroadcastInput would enforce
    if (transposedWeightsShape.size() != 4 || transposedActivationShape.size() != 4) {
        return mlir::failure();
    }

    // Check if transposed weights shape would satisfy verifyAndBroadcastInput constraints
    // For weights: [N, C, H, W] should be [1, ?, 1, 1] after transpose
    if (transposedWeightsShape[0] != 1 || transposedWeightsShape[2] != 1 || transposedWeightsShape[3] != 1) {
        return mlir::failure();
    }

    // Check channel compatibility between transposed weights and activation
    // transposedWeightsShape[1] is the channel dimension after transpose
    // transposedActivationShape[1] is the channel dimension after transpose
    if (transposedWeightsShape[1] != transposedActivationShape[1] && transposedWeightsShape[1] != 1) {
        return mlir::failure();
    }

    return mlir::success();
}

mlir::LogicalResult checkIfShapesAreBroadcastable(ArrayRef<int64_t> shape1, ArrayRef<int64_t> shape2,
                                                  IE::AutoBroadcastType broadcastType) {
    if (broadcastType == IE::AutoBroadcastType::NONE_OR_EXPLICIT) {
        if (shape1 != shape2) {
            return mlir::failure();
        }

        return mlir::success();
    } else if (broadcastType == IE::AutoBroadcastType::NUMPY) {
        auto in1ShapeIter = shape1.rbegin();
        auto in2ShapeIter = shape2.rbegin();
        while (in1ShapeIter != shape1.rend() && in2ShapeIter != shape2.rend()) {
            if (*in1ShapeIter != 1 && *in2ShapeIter != 1 && *in1ShapeIter != *in2ShapeIter) {
                return mlir::failure();
            }

            if (in1ShapeIter != shape1.rend()) {
                ++in1ShapeIter;
            }
            if (in2ShapeIter != shape2.rend()) {
                ++in2ShapeIter;
            }
        }

        return mlir::success();
    }

    return mlir::failure();
}

bool checkIfNeedToCloneOpChain(mlir::Operation* chainOp, ShapeRef dataConstOpShape) {
    for (auto* userOp : chainOp->getUsers()) {
        auto outputShape = getShape(userOp->getResult(0));
        bool needsClone = false;

        if (userOp->hasAttr("auto_broadcast")) {
            static const auto N = Dims4D::Act::N;
            static const auto C = Dims4D::Act::C;
            static const auto H = Dims4D::Act::H;
            static const auto W = Dims4D::Act::W;

            auto broadcastType =
                    mlir::dyn_cast<vpux::IE::AutoBroadcastTypeAttr>(userOp->getAttr("auto_broadcast")).getValue();

            SmallVector<int64_t> shape1 = {outputShape[N], outputShape[C], outputShape[H], outputShape[W]};
            SmallVector<int64_t> shape2 = {dataConstOpShape[N], dataConstOpShape[C], dataConstOpShape[H],
                                           dataConstOpShape[W]};

            if (mlir::failed(checkIfShapesAreBroadcastable(shape1, shape2, broadcastType))) {
                return true;
            }
        } else if (!mlir::isa<IE::ReshapeOp>(userOp) && outputShape != dataConstOpShape) {
            return true;
        }

        if (mlir::isa<IE::ReshapeOp, IE::FakeQuantizeOp>(userOp)) {
            needsClone = checkIfNeedToCloneOpChain(userOp, dataConstOpShape);
        }

        if (needsClone) {
            return true;
        }
    }
    return false;
}

mlir::LogicalResult verifyAndBroadcastInput(mlir::Location loc, mlir::Value& input, Shape inputShape, Shape outputShape,
                                            mlir::Value& newInput, mlir::PatternRewriter& rewriter,
                                            mlir::Value activationInput = nullptr,
                                            mlir::Value* transposedActivation = nullptr,
                                            const std::optional<TransposeInfo>& transposeInfoValue = std::nullopt) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    Shape newInputShape(std::move(inputShape));
    Shape newOutputShape(std::move(outputShape));

    // Handle spatial transpose if needed
    if (transposeInfoValue.has_value() && activationInput && transposedActivation) {
        auto transposeInfo = transposeInfoValue.value();
        // For H/W-expanded weights requiring transpose, activation input must be constant or TransposeOp
        auto preTransposeOp = activationInput.getDefiningOp<IE::TransposeOp>();
        if (mlir::failed(IE::getConstParentOp(activationInput)) && preTransposeOp == nullptr) {
            return mlir::failure();
        }

        // Validate that transpose operations would be feasible before creating them
        auto activationShape = getShape(activationInput);
        if (mlir::failed(validateTransposeOperations(newInputShape, activationShape, transposeInfo))) {
            return mlir::failure();
        }

        // If transpose is needed, compute the transposed shapes for validation
        SmallVector<int64_t> transposedActivationShape = computeTransposedShape(activationShape, transposeInfo);
        SmallVector<int64_t> transposedWeightsShape = computeTransposedShape(newInputShape, transposeInfo);

        newInputShape = Shape(transposedWeightsShape);
        newOutputShape = Shape(transposedActivationShape);
    }

    if (newOutputShape.size() != 4 || newInputShape.size() != 4) {
        return mlir::failure();
    }
    if (newInputShape[N] != 1 || newInputShape[H] != 1 || newInputShape[W] != 1) {
        return mlir::failure();
    }

    if (newInputShape[C] != newOutputShape[C] && newInputShape[C] != 1) {
        return mlir::failure();
    }

    // Broadcast scalar for all channels
    if (newInputShape[C] != newOutputShape[C] && newInputShape[C] == 1) {
        SmallVector<mlir::Operation*> opsVec;
        Const::DeclareOp input2Const = nullptr;
        // Convert [Const] -> [optional several Reshapes]-> [optional FQ] -> [optional several Reshapes] ->
        // [Multiply/Add] case to scaleShift
        mlir::Operation* operation = input.getDefiningOp();
        if (operation == nullptr) {
            return mlir::failure();
        }
        while (operation && mlir::isa<IE::ReshapeOp, IE::FakeQuantizeOp, Const::DeclareOp>(operation)) {
            if (mlir::isa<IE::ReshapeOp, IE::FakeQuantizeOp>(operation)) {
                opsVec.insert(opsVec.begin(), operation);
                operation = operation->getOperand(0).getDefiningOp();
                continue;  // Continue searching for Const::DeclareOp
            }

            if (mlir::isa<Const::DeclareOp>(operation)) {
                input2Const = mlir::dyn_cast_or_null<Const::DeclareOp>(operation);
                break;
            }
        }

        // Const input can not be found
        if (input2Const == nullptr) {
            return mlir::failure();
        }

        Const::ContentAttr dataAttr = input2Const.transformContentAttr().broadcast(C, newOutputShape[C]).get();

        if (dataAttr == nullptr) {
            return mlir::failure();
        }

        auto dataConstOp = rewriter.create<Const::DeclareOp>(loc, dataAttr.getType(), std::move(dataAttr));
        auto dataConstOpShape = getShape(dataConstOp.getOutput());

        bool needToCloneOpChain = checkIfNeedToCloneOpChain(input2Const, dataConstOpShape);

        if (opsVec.size() == 0) {
            // [Const]->[Multiply/Add] case
            if (needToCloneOpChain) {
                newInput = dataConstOp.getOutput();
            } else {
                input = dataConstOp.getOutput();
                newInput = input;
            }
        } else {
            // [Const] -> [several Reshapes]-> [FQ] -> [several Reshapes] -> [Multiply/Add] case
            if (needToCloneOpChain) {
                SmallVector<mlir::Operation*> opsVecCopy;
                for (auto op : opsVec) {
                    auto copyOp = rewriter.clone(*op);
                    copyOp->setLoc(appendLoc(loc, "copy_scale_shift"));
                    opsVecCopy.push_back(copyOp);
                }

                opsVecCopy.front()->getOpOperand(0).set(dataConstOp.getOutput());
                for (auto op : opsVecCopy) {
                    inferReturnTypes(op, InferShapedTypeMode::SHAPE);
                }

                newInput = opsVecCopy.front()->getResult(0);
            } else {
                opsVec.front()->getOpOperand(0).set(dataConstOp.getOutput());
                for (auto op : opsVec) {
                    inferReturnTypes(op, InferShapedTypeMode::SHAPE);
                }
                newInput = input;
            }
        }
    }

    // Create transpose operations if needed (after shape validation and broadcasting)
    if (transposeInfoValue.has_value() && activationInput && transposedActivation) {
        auto transposeInfo = transposeInfoValue.value();
        auto transposeOrderAttr = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(transposeInfo.permutation, rewriter.getContext()));

        // Lambda function to create transpose operations
        auto createTransposeOp = [&](mlir::Value inputValue, mlir::Location opLoc) -> mlir::Value {
            auto inputType = mlir::cast<mlir::ShapedType>(inputValue.getType());
            auto transposedShape = computeTransposedShape(getShape(inputValue), transposeInfo);
            auto outputType = inputType.clone(transposedShape);

            auto transposeOp =
                    rewriter.create<IE::TransposeOp>(opLoc, outputType, inputValue, nullptr, transposeOrderAttr);
            return transposeOp.getResult();
        };

        // Create transpose operations using the lambda
        *transposedActivation = createTransposeOp(activationInput, appendLoc(loc, "trans_activation"));
        input = createTransposeOp(input, appendLoc(loc, "trans_weights"));
    }

    return mlir::success();
}

//
// ConvertBiasToScaleShift
//

template <typename BiasTypeOp>
class ConvertBiasToScaleShift final : public mlir::OpRewritePattern<BiasTypeOp> {
public:
    ConvertBiasToScaleShift<BiasTypeOp>(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<BiasTypeOp>(ctx, benefit), _log(log) {
        this->setDebugName("ConvertBiasToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(BiasTypeOp addOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

template <typename BiasTypeOp>
mlir::LogicalResult ConvertBiasToScaleShift<BiasTypeOp>::matchAndRewrite(BiasTypeOp biasOp,
                                                                         mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", biasOp->getName(), biasOp->getLoc());
    auto inElemType = mlir::cast<vpux::NDTypeInterface>(biasOp.getInput2().getType()).getElementType();
    auto outElemType = mlir::cast<vpux::NDTypeInterface>(biasOp.getOutput().getType()).getElementType();

    // from the ops defination, scale shift can only support F16
    if (!(inElemType.isF16())) {
        _log.trace("Could not convert to scale shift due to input date type is not FP16");
        return mlir::failure();
    }

    if (inElemType != outElemType) {
        _log.nest().trace("op {0} input and output types are not matching", biasOp->getName());
        return mlir::failure();
    }

    bool lhsIsActivation = mlir::failed(IE::getConstParentOp(biasOp.getInput1()));
    mlir::Value activationInput = lhsIsActivation ? biasOp.getInput1() : biasOp.getInput2();
    mlir::Value biasInput = lhsIsActivation ? biasOp.getInput2() : biasOp.getInput1();

    auto findBiasConst = IE::getConstParentOp(biasInput);
    if (mlir::failed(findBiasConst)) {
        _log.nest().trace("op {0} input is not constant", biasOp->getName());
        return mlir::failure();
    }

    if (mlir::isa<IE::SubtractOp>(biasOp) && !lhsIsActivation) {
        _log.nest().trace("op {0} activation is not the first input", biasOp->getName());
        return mlir::failure();
    }

    auto mulOutShape = getShape(biasOp.getOutput());
    auto biasesShape = getShape(biasInput);

    auto newInput = biasInput;
    if (verifyAndBroadcastInput(biasOp.getLoc(), biasInput, Shape(biasesShape), Shape(mulOutShape), newInput, rewriter,
                                nullptr, nullptr, std::nullopt)
                .failed()) {
        _log.nest().trace("op {0} input cannot be broadcast", biasOp->getName());
        return mlir::failure();
    }

    findBiasConst = IE::getConstParentOp(newInput);
    auto biasConst = findBiasConst.value();

    // Convert:
    //
    // Tensor              Const
    //    |                  |
    //    |               Negative        Tensor              Const
    //    |                  |               |                  |
    //     \______AddOp______/                \______SubOp______/
    //              |                                  |
    //
    // To:
    //
    // Tensor             NewConst
    //    |                  |
    //    |                  |
    //    |                  |
    //     \___ScaleShift___/
    //              |

    if (mlir::isa<IE::NegativeOp>(newInput.getDefiningOp()) || mlir::isa<IE::SubtractOp>(biasOp)) {
        auto negativeConstAttr = biasConst.transformContentAttr().rescale(-1.0).get();
        newInput = rewriter.create<Const::DeclareOp>(takeOpLoc(biasOp, "bias_in"), biasConst.getType(),
                                                     std::move(negativeConstAttr))
                           .getOutput();
    }

    _log.nest().trace("replaced op {0} with ScaleShift", biasOp->getName());
    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(biasOp, biasOp.getType(), activationInput, nullptr, newInput);

    return mlir::success();
}

//
// ConvertMultiplyToScaleShift
//

class ConvertMultiplyToScaleShift : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    ConvertMultiplyToScaleShift(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefit), _log(log) {
        this->setDebugName("ConvertMultiplyToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

protected:
    Logger _log;
};

bool isBeneficialToConvertMultiplyToScaleShift(ShapeRef activationShape, ShapeRef weightsShape, ShapeRef outputShape,
                                               const IE::MultiplyOp& mulOp, const Logger& log) {
    const int64_t dimCShape = outputShape[Dim(Dims4D::Act::C)];
    if (dimCShape <= VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        log.trace("Operations with C dimension <= 8192 can be converted to ScaleShift");
        return true;
    }

    if (config::getArch(mulOp) <= config::ArchKind::NPU40XX) {
        log.trace("Operations with C dimension > 8192 on NPU40xx and older is faster on SHAVE");
        return false;
    }

    // Operations benefit from running on DPU when channel dimension size is less than
    // 2x(experimental value) the standard limit
    // E-171794 will introduce a comprehensive solution for choosing between different executors
    constexpr double DPU_BENEFIT_FACTOR = 2;
    const bool isBenefitOnDPU =
            dimCShape < static_cast<int64_t>(VPU::NCEInvariant::VPU_DIMENSION_LIMIT * DPU_BENEFIT_FACTOR);
    // Operations that do not need to be broadcasted can be decided to execute on DPU(NCEEltwise) or
    // SHAVE(VPU.Multiply) in later passes
    const bool needBroadcast = activationShape != weightsShape;
    if (needBroadcast && isBenefitOnDPU) {
        log.trace("Operations that need to be broadcasted with C dimension > 8192 can be converted to ScaleShift");
        return true;
    }

    return false;
}

mlir::LogicalResult ConvertMultiplyToScaleShift::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", mulOp->getName(), mulOp->getLoc());
    auto mulOutShape = getShape(mulOp.getOutput());
    auto lhsShape = getShape(mulOp.getInput1());
    auto rhsShape = getShape(mulOp.getInput2());

    // Choose the input that matches the output shape as activation
    bool lhsIsActivation = (lhsShape == mulOutShape);
    bool rhsIsActivation = (rhsShape == mulOutShape);

    // If neither input matches output shape exactly, or both match, fall back to original logic
    if (!lhsIsActivation && !rhsIsActivation) {
        return mlir::failure();
    }

    // From the ops definition, scale shift can only support F16
    const auto lhsElementType = mlir::cast<mlir::ShapedType>(mulOp.getInput1().getType()).getElementType();
    if (!lhsElementType.isF16()) {
        _log.trace("Could not convert to scale shift due to input data type is not FP16");
        return mlir::failure();
    }

    mlir::Value activationInput = lhsIsActivation ? mulOp.getInput1() : mulOp.getInput2();
    mlir::Value weightsInput = lhsIsActivation ? mulOp.getInput2() : mulOp.getInput1();

    auto weightsShape = getShape(weightsInput);
    auto activationShape = getShape(activationInput);

    // Activation shape and scaleShift output shape should be consistent
    if (activationShape != mulOutShape) {
        return mlir::failure();
    }

    if (!isBeneficialToConvertMultiplyToScaleShift(activationShape, weightsShape, mulOutShape, mulOp, _log)) {
        return mlir::failure();
    }

    auto newInput = weightsInput;
    mlir::Value transposedActivation = activationInput;
    std::optional<TransposeInfo> transposeInfo = needsSpatialTranspose(weightsShape);

    // Verify and broadcast input, handling transpose if needed
    if (verifyAndBroadcastInput(mulOp.getLoc(), newInput, Shape(weightsShape), Shape(mulOutShape), newInput, rewriter,
                                activationInput, &transposedActivation, transposeInfo)
                .failed()) {
        return mlir::failure();
    }

    // Create ScaleShift operation
    _log.nest().trace("Replacing {0} with ScaleShift", mulOp->getName());
    auto scaleShiftOp = rewriter.create<IE::ScaleShiftOp>(
            takeOpLoc(mulOp, "as_scaleshift"), transposedActivation.getType(), transposedActivation, newInput, nullptr);

    // Apply inverse transpose if we did spatial transpose earlier
    mlir::Value finalOutput = scaleShiftOp.getOutput();
    if (transposeInfo) {
        auto inverseTransposeOrderAttr = mlir::AffineMapAttr::get(
                mlir::AffineMap::getPermutationMap(transposeInfo->inverse, rewriter.getContext()));

        auto inverseTransposeOp =
                rewriter.create<IE::TransposeOp>(appendLoc(mulOp.getLoc(), "restore_layout"), mulOp.getType(),
                                                 finalOutput, nullptr, inverseTransposeOrderAttr);
        finalOutput = inverseTransposeOp.getOutput();

        _log.trace("Applied inverse transpose to restore original layout");
    }

    rewriter.replaceOp(mulOp, finalOutput);
    return mlir::success();
}

//
// FoldMultiplyHWSplatWeights
//

class FoldMultiplyHWSplatWeights : public mlir::OpRewritePattern<IE::MultiplyOp> {
public:
    FoldMultiplyHWSplatWeights(mlir::MLIRContext* ctx, mlir::PatternBenefit benefit, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx, benefit), _log(log) {
        this->setDebugName("FoldMultiplyHWSplatWeights");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

protected:
    Logger _log;
};

mlir::LogicalResult FoldMultiplyHWSplatWeights::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", mulOp->getName(), mulOp->getLoc());
    const auto lhsType = mlir::cast<mlir::ShapedType>(mulOp.getInput1().getType());
    const auto outShapeRes = mlir::cast<mlir::ShapedType>(mulOp.getOutput().getType());

    bool lhsIsActivation = (lhsType == outShapeRes);
    mlir::Value activationInput = lhsIsActivation ? mulOp.getInput1() : mulOp.getInput2();
    mlir::Value weightsInput = lhsIsActivation ? mulOp.getInput2() : mulOp.getInput1();

    auto mulOutShape = getShape(mulOp.getOutput());
    auto weightsShape = getShape(weightsInput);

    // Activation shape and scaleShift output shape should be consistent
    if (getShape(activationInput) != mulOutShape) {
        return mlir::failure();
    }

    const int64_t rank4D = 4;
    if (mulOutShape.size() != rank4D || weightsShape.size() != rank4D) {
        return mlir::failure();
    }

    // Handle the below weights shape patterns:
    // <1x1xHx1> isSplat -> <1x1x1x1>
    // <1x1x1xW> isSplat -> <1x1x1x1>
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;
    if (!(weightsShape[N] == 1 && weightsShape[C] == 1 &&
          ((weightsShape[W] == 1 && weightsShape[H] != 1) || (weightsShape[H] == 1 && weightsShape[W] != 1)))) {
        return mlir::failure();
    }

    auto weightsConstOp = mlir::dyn_cast_or_null<Const::DeclareOp>(weightsInput.getDefiningOp());
    if (weightsConstOp == nullptr) {
        return mlir::failure();
    }

    const auto& constAttr = weightsConstOp.getContentAttr();
    if (!constAttr.isSplat()) {
        return mlir::failure();
    }

    const auto offset = Shape(weightsShape.size(), 0);
    const auto shape = Shape(weightsShape.size(), 1);
    Const::ContentAttr newConstAttr = constAttr.transform().subview(offset, shape).get();
    if (newConstAttr == nullptr) {
        return mlir::failure();
    }

    // Create new weights Const with shape 1x1x1x1
    rewriter.setInsertionPoint(mulOp);
    auto newWeightsInput =
            rewriter.create<Const::DeclareOp>(mulOp.getLoc(), newConstAttr.getType(), std::move(newConstAttr))
                    .getOutput();

    weightsInput.replaceUsesWithIf(newWeightsInput, [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == mulOp;
    });

    return mlir::success();
}

//
// ConvertToScaleShiftPass
//

class ConvertToScaleShiftPass final : public IE::impl::ConvertToScaleShiftBase<ConvertToScaleShiftPass> {
public:
    explicit ConvertToScaleShiftPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void ConvertToScaleShiftPass::safeRunOnFunc() {
    auto& ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<FoldMultiplyHWSplatWeights>(&ctx, benefitLevels[0], _log);
    patterns.add<ConvertBiasToScaleShift<IE::AddOp>>(&ctx, benefitLevels[1], _log);
    patterns.add<ConvertBiasToScaleShift<IE::SubtractOp>>(&ctx, benefitLevels[1], _log);
    patterns.add<ConvertMultiplyToScaleShift>(&ctx, benefitLevels[1], _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertToScaleShiftPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createConvertToScaleShiftPass(Logger log) {
    return std::make_unique<ConvertToScaleShiftPass>(log);
}
