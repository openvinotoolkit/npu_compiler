//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_INPUTQUANTIZATIONRESTORATION
#define GEN_PASS_DEF_INPUTQUANTIZATIONRESTORATION
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int levels = QuantizationLevels::QUANT_LEVELS_8BIT;

//
// InputQuantizationRestoration
//

class InputQuantizationRestoration final :
        public IE::impl::InputQuantizationRestorationBase<InputQuantizationRestoration> {
public:
    explicit InputQuantizationRestoration(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// InputQuantizationRestoration
//

using FQParametersType = std::tuple<mlir::Value,  // input low
                                    mlir::Value,  // input high
                                    mlir::Value,  // output low
                                    mlir::Value   // output high
                                    >;

FQParametersType createQuantizationConstants(mlir::PatternRewriter& rewriter, float scale, float zeroPoint,
                                             mlir::Location loc, mlir::ArrayRef<int64_t> shape) {
    // Input low and high should be the range of U8 data type
    float inputLowValue = static_cast<float>(std::numeric_limits<uint8_t>::min());
    float inputHighValue = static_cast<float>(std::numeric_limits<uint8_t>::max());
    // Calculate output low and high values using the the scale and zeropoint
    float outputLowValue = -scale * zeroPoint;
    float outputHighValue = scale * ((levels - 1) - zeroPoint);

    // Create a shape vector of the same rank, all ones
    std::vector<int64_t> onesShape(shape.size(), 1);

    // Use this shape for the constant tensor type
    auto tensorType = mlir::RankedTensorType::get(onesShape, rewriter.getF32Type());

    auto inputLow = Const::createConst(rewriter, loc, tensorType, ArrayRef(inputLowValue));
    auto inputHigh = Const::createConst(rewriter, loc, tensorType, ArrayRef(inputHighValue));
    auto outputLow = Const::createConst(rewriter, loc, tensorType, ArrayRef(outputLowValue));
    auto outputHigh = Const::createConst(rewriter, loc, tensorType, ArrayRef(outputHighValue));

    return std::make_tuple(inputLow, inputHigh, outputLow, outputHigh);
}

class InputFakeQuantizeInsertionPattern final : public mlir::OpRewritePattern<IE::ConvertOp> {
public:
    InputFakeQuantizeInsertionPattern(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::ConvertOp>(ctx), _log(log) {
    }

    mlir::Operation* getLastPreProcOp(mlir::Operation* currOp) const {
        auto nextOp = currOp;
        while (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp>(nextOp) && nextOp->getResult(0).hasOneUse()) {
            currOp = nextOp;
            nextOp = *currOp->getResult(0).getUsers().begin();
        }
        return currOp;
    }

    mlir::Operation* getFirstInputPreProcOp(mlir::Operation* currOp) const {
        auto nextOp = currOp;
        while (mlir::isa_and_nonnull<IE::ConvertOp, IE::TransposeOp>(nextOp)) {
            currOp = nextOp;
            nextOp = currOp->getOperand(0).getDefiningOp();
        }
        return currOp;
    }

    mlir::LogicalResult matchAndRewrite(IE::ConvertOp convertOp, mlir::PatternRewriter& rewriter) const final {
        auto input = convertOp.getInput();
        auto firstPreProcOp = getFirstInputPreProcOp(convertOp);
        auto firstPreProcOpInput = firstPreProcOp->getOperand(0);

        if (!mlir::isa<mlir::BlockArgument>(firstPreProcOpInput)) {
            return mlir::failure();
        }

        if (mlir::cast<vpux::NDTypeInterface>(input.getType()).getElementType() !=
            vpux::getUInt8Type(rewriter.getContext())) {
            return mlir::failure();
        }

        auto isInputQuantizationUsecase = [this](mlir::Operation* op) -> bool {
            for (auto user : op->getResult(0).getUsers()) {
                if (!mlir::isa_and_nonnull<IE::LayerWithPostOpInterface>(user)) {
                    return false;
                }

                if (op->getResult(0) == user->getOperand(0)) {
                    // First argument is activation, this case is always allowed
                    continue;
                }

                if (!user->hasTrait<IE::EltwiseOp>()) {
                    // This is DPU operation with weights quantization pattern, skip it
                    return false;
                }

                if (user->getNumOperands() == 2) {
                    // Check that both operands have the same shape
                    auto operand1Shape = getShape(user->getOperand(0)).raw();
                    auto operand2Shape = getShape(user->getOperand(1)).raw();
                    // It is considered input quantization pattern for second input of Eltwise op only if the
                    // shape of the two operands is equal
                    if (!operand1Shape.equals(operand2Shape)) {
                        _log.trace("No quantization pattern of second Eltwise input because operands have "
                                   "different shapes.");
                        return false;
                    }
                }
            }
            return true;
        };

        auto lastPreProcOp = getLastPreProcOp(convertOp);
        auto inputShape = mlir::cast<mlir::RankedTensorType>(lastPreProcOp->getResult(0).getType()).getShape();

        // Analyze users of the last preprocessing operation
        for (auto* user : lastPreProcOp->getResult(0).getUsers()) {
            // Subtract pattern
            // Case 1: lastPreProcOp -> Subtract
            auto subtractOp = mlir::dyn_cast<IE::SubtractOp>(user);
            if (subtractOp != nullptr) {
                auto zeroPointOr = getZeroPoint(subtractOp.getInput2());
                if (mlir::failed(zeroPointOr)) {
                    return mlir::failure();
                }
                float zeroPoint = zeroPointOr.value();

                // Check Multiply users
                IE::MultiplyOp multiplyOp = nullptr;
                for (auto* subUser : subtractOp->getResult(0).getUsers()) {
                    if (auto mulUser = mlir::dyn_cast<IE::MultiplyOp>(subUser)) {
                        multiplyOp = mulUser;
                    }
                }
                if (subtractOp->getResult(0).hasOneUse() && multiplyOp != nullptr &&
                    isInputQuantizationUsecase(multiplyOp)) {
                    // Case 3: lastPreProcOp -> Subtract -> Multiply, Subtract has only one use (Multiply)
                    auto scaleValueOr = extractScaleValue(multiplyOp, subtractOp.getResult());
                    if (mlir::failed(scaleValueOr)) {
                        return mlir::failure();
                    }
                    float scaleValue = scaleValueOr.value();
                    createAndReplaceFakeQuantize(rewriter, multiplyOp->getLoc(), multiplyOp,
                                                 lastPreProcOp->getResult(0), scaleValue, zeroPoint, inputShape);
                    return mlir::success();
                } else if (isInputQuantizationUsecase(subtractOp)) {
                    // Case 1 fallback: lastPreProcOp -> Subtract, Subtract has multiple uses or not only Multiply
                    createAndReplaceFakeQuantize(rewriter, subtractOp->getLoc(), subtractOp,
                                                 lastPreProcOp->getResult(0), 1.0f, zeroPoint, inputShape);
                    return mlir::success();
                }
            }
            // Case 2: lastPreProcOp -> Multiply
            if (auto multiplyOp = mlir::dyn_cast<IE::MultiplyOp>(user)) {
                auto scaleValueOr = extractScaleValue(multiplyOp, lastPreProcOp->getResult(0));
                if (mlir::failed(scaleValueOr) || !isInputQuantizationUsecase(multiplyOp)) {
                    return mlir::failure();
                }
                float scaleValue = scaleValueOr.value();
                float zeroPoint = 0.0f;
                createAndReplaceFakeQuantize(rewriter, multiplyOp->getLoc(), multiplyOp, lastPreProcOp->getResult(0),
                                             scaleValue, zeroPoint, inputShape);
                return mlir::success();
            }
        }

        return mlir::failure();
    }

private:
    Logger _log;

    void createAndReplaceFakeQuantize(mlir::PatternRewriter& rewriter, mlir::Location loc, mlir::Operation* toReplace,
                                      mlir::Value input, float scale, float zeroPoint,
                                      mlir::ArrayRef<int64_t> inputShape) const {
        auto [inputLow, inputHigh, outputLow, outputHigh] =
                createQuantizationConstants(rewriter, scale, zeroPoint, loc, inputShape);
        rewriter.setInsertionPointAfter(input.getDefiningOp());
        auto newOp = rewriter.replaceOpWithNewOp<IE::FakeQuantizeOp>(
                toReplace, input, inputLow, inputHigh, outputLow, outputHigh, rewriter.getI64IntegerAttr(levels),
                nullptr,
                vpux::IE::AutoBroadcastTypeAttr::get(rewriter.getContext(), vpux::IE::AutoBroadcastType::NUMPY));
        extendOpLoc(newOp, "as_fq");
    }

    mlir::FailureOr<float> extractScaleValue(IE::MultiplyOp mul, mlir::Value referenceInput) const {
        auto scaleOperand = getScaleOperand(mul, referenceInput);
        if (!isValidScale(scaleOperand)) {
            return mlir::failure();
        }
        auto scaleConstOp = scaleOperand.getDefiningOp<Const::DeclareOp>();
        auto scaleValueOr = vpux::Const::getSplatValue<float>(scaleConstOp);
        if (mlir::failed(scaleValueOr)) {
            return mlir::failure();
        }
        return scaleValueOr.value();
    }

    mlir::Value getScaleOperand(IE::MultiplyOp mul, mlir::Value referenceInput) const {
        return mul.getOperand(0) == referenceInput ? mul.getOperand(1) : mul.getOperand(0);
    }

    bool isValidScale(mlir::Value val) const {
        auto type = mlir::dyn_cast<mlir::RankedTensorType>(val.getType());
        if (!type) {
            return false;
        }
        auto shape = type.getShape();
        for (size_t i = 1; i < shape.size(); ++i) {
            if (shape[i] != 1) {
                return false;
            }
        }
        auto constOp = val.getDefiningOp<Const::DeclareOp>();
        if (!constOp) {
            return false;
        }
        auto splat = vpux::Const::getSplatValue<float>(constOp);
        return mlir::succeeded(splat);
    }

    mlir::FailureOr<float> getZeroPoint(mlir::Value val) const {
        auto constOp = val.getDefiningOp<Const::DeclareOp>();
        if (!constOp) {
            _log.warning("Zero point constant not found for value: {0}", val);
            return mlir::failure();
        }
        auto valueOr = vpux::Const::getSplatValue<float>(constOp);
        if (mlir::failed(valueOr)) {
            _log.warning("Zero point constant is not a splat or failed to extract for value: {0}", val);
            return mlir::failure();
        }
        return valueOr.value();
    }
};

void InputQuantizationRestoration::safeRunOnFunc() {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(func.getContext());
    patterns.add<InputFakeQuantizeInsertionPattern>(func.getContext(), _log);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::IE::createInputQuantizationRestorationPass(Logger log) {
    return std::make_unique<InputQuantizationRestoration>(log);
}
