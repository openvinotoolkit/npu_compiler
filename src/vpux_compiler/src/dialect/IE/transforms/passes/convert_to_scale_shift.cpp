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

// To explicitly control the patterns exec order to assure dependency
// benefitLevels[0] is highest benefit level and represent the relative pattern is the first one to run
const uint32_t levelCount = 2;
SmallVector<mlir::PatternBenefit> benefitLevels = getBenefitLevels(levelCount);

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

mlir::LogicalResult verifyAndBroadcastInput(mlir::Location loc, mlir::Value& input, vpux::ShapeRef inputShape,
                                            vpux::ShapeRef outputShape, mlir::Value& newInput,
                                            mlir::PatternRewriter& rewriter) {
    static const auto N = Dims4D::Act::N;
    static const auto C = Dims4D::Act::C;
    static const auto H = Dims4D::Act::H;
    static const auto W = Dims4D::Act::W;

    if (outputShape.size() != 4 || inputShape.size() != 4) {
        return mlir::failure();
    }
    if (inputShape[N] != 1 || inputShape[H] != 1 || inputShape[W] != 1) {
        return mlir::failure();
    }

    if (inputShape[C] != outputShape[C] && inputShape[C] != 1) {
        return mlir::failure();
    }

    // Broadcast scalar for all channels
    if (inputShape[C] != outputShape[C] && inputShape[C] == 1) {
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

        Const::ContentAttr dataAttr = input2Const.transformContentAttr().broadcast(C, outputShape[C]).get();

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
    if (verifyAndBroadcastInput(biasOp.getLoc(), biasInput, biasesShape, mulOutShape, newInput, rewriter).failed()) {
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
    const auto lhsType = mlir::cast<mlir::ShapedType>(mulOp.getInput1().getType());
    const auto outShapeRes = mlir::cast<mlir::ShapedType>(mulOp.getOutput().getType());

    // From the ops definition, scale shift can only support F16
    const auto lhsElementType = lhsType.getElementType();
    if (!(lhsElementType.isF16())) {
        _log.trace("Could not convert to scale shift due to input data type is not FP16");
        return mlir::failure();
    }

    bool lhsIsActivation = (lhsType == outShapeRes);
    mlir::Value activationInput = lhsIsActivation ? mulOp.getInput1() : mulOp.getInput2();
    mlir::Value weightsInput = lhsIsActivation ? mulOp.getInput2() : mulOp.getInput1();

    auto mulOutShape = getShape(mulOp.getOutput());
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
    if (verifyAndBroadcastInput(mulOp.getLoc(), weightsInput, weightsShape, mulOutShape, newInput, rewriter).failed()) {
        return mlir::failure();
    }

    // Convert:
    //
    // Tensor                 Const
    //    |                     |
    //    |                     |
    //    |                     |
    //     \_____MultiplyOp____/
    //               |
    //
    // To:
    //
    // Tensor             NewConst
    //    |                  |
    //    |                  |
    //    |                  |
    //     \___ScaleShift___/
    //              |

    _log.nest().trace("replaced op {0} with ScaleShift", mulOp->getName());
    rewriter.replaceOpWithNewOp<IE::ScaleShiftOp>(mulOp, mulOp.getType(), activationInput, newInput, nullptr);

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
