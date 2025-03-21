//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"

#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_CONVERTTOSCALESHIFT
#define GEN_PASS_DEF_CONVERTTOSCALESHIFT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

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

            auto broadcastType = userOp->getAttr("auto_broadcast").dyn_cast<IE::AutoBroadcastTypeAttr>().getValue();

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
    ConvertBiasToScaleShift<BiasTypeOp>(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<BiasTypeOp>(ctx), _log(log) {
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
    auto inElemType = biasOp.getInput2().getType().template cast<vpux::NDTypeInterface>().getElementType();
    auto outElemType = biasOp.getOutput().getType().template cast<vpux::NDTypeInterface>().getElementType();

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
    ConvertMultiplyToScaleShift(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::MultiplyOp>(ctx), _log(log) {
        this->setDebugName("ConvertMultiplyToScaleShift");
    }

    mlir::LogicalResult matchAndRewrite(IE::MultiplyOp mulOp, mlir::PatternRewriter& rewriter) const final;

protected:
    Logger _log;
};

mlir::LogicalResult ConvertMultiplyToScaleShift::matchAndRewrite(IE::MultiplyOp mulOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.trace("Got op {0} at {1}", mulOp->getName(), mulOp->getLoc());
    const auto lhsType = mulOp.getInput1().getType().cast<mlir::ShapedType>();
    const auto outShapeRes = mulOp.getOutput().getType().cast<mlir::ShapedType>();

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

    // Activation shape and scaleShift output shape should be consistent
    if (getShape(activationInput) != mulOutShape) {
        return mlir::failure();
    }

    if (mulOutShape[Dim(Dims4D::Act::C)] > VPU::NCEInvariant::VPU_DIMENSION_LIMIT) {
        _log.trace("Multiply with C Dim > 8192 will not be converted to ScaleShift since it is faster on Shave.");
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
    patterns.add<ConvertBiasToScaleShift<IE::AddOp>>(&ctx, _log);
    patterns.add<ConvertBiasToScaleShift<IE::SubtractOp>>(&ctx, _log);
    patterns.add<ConvertMultiplyToScaleShift>(&ctx, _log);

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
