//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_quant.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

using namespace vpux;
using namespace IE;

namespace {
// Helper function to check if operation has a PostOp attached
inline bool hasPPE(mlir::Operation* operation) {
    if (auto layerWithPostOp = mlir::dyn_cast_or_null<IE::LayerWithPostOpInterface>(operation)) {
        return layerWithPostOp.hasPPE();
    }
    return false;
}
}  // namespace

template <typename ConcreteOp>
mlir::LogicalResult IE::findQuantizeOrQuantizedNCE(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                                   mlir::Value eltwiseInput,
                                                   SmallVector<mlir::Operation*>& eltwiseToQuantizeOps, Logger log) {
    auto verifyInput = [origOp, &rewriter](mlir::Operation* operation) {
        if (operation == nullptr) {
            return matchFailed(rewriter, origOp, "Producer is a block argument for {0} at {1}", origOp->getName(),
                               origOp->getLoc());
        }
        if (!mlir::isa<IE::ElemTypeInfoOpInterface, IE::LayerWithPostOpInterface, IE::QuantizeOp>(operation)) {
            return matchFailed(rewriter, origOp,
                               "Ancestor {0} at {1} is neither FakeQuantize-agnostic, NCE, nor Quantize operation",
                               operation->getName(), operation->getLoc());
        }
        VPUX_THROW_UNLESS(!operation->use_empty(),
                          "Expact operation always to have uses because the loop just came from one of it's uses; "
                          "the loop is walking up the calling chain");
        if (!operation->hasOneUse()) {
            // TODO: Should use hasOneUser when it's available in MLIR:
            // https://llvm.org/doxygen/classllvm_1_1Value.html#a2e987c6af902aad6baa39bd5b7ef322c
            const auto hasOneUser =
                    std::equal(++operation->user_begin(), operation->user_end(), operation->user_begin());
            if (!hasOneUser) {
                return matchFailed(rewriter, origOp, "Ancestor has more than one consumer for {0} at {1} ",
                                   operation->getName(), operation->getLoc());
            }
        }
        if (mlir::isa<IE::ElemTypeInfoOpInterface>(operation) && operation->getNumOperands() > 1) {
            return matchFailed(rewriter, origOp,
                               "ElemTypeInfoOpInterface Ancestor has more than one input for {0} at {1}",
                               operation->getName(), operation->getLoc());
        }

        // Block fusion if operation has PPE
        if (::hasPPE(operation)) {
            return matchFailed(rewriter, origOp, "Ancestor {0} at {1} has PPE, cannot fuse quantization through it",
                               operation->getName(), operation->getLoc());
        }

        return mlir::success();
    };

    do {
        mlir::Operation* input = [eltwiseInput, eltwiseToQuantizeOps = ArrayRef(eltwiseToQuantizeOps)]() {
            if (eltwiseToQuantizeOps.empty()) {
                return eltwiseInput.getDefiningOp();
            } else {
                return eltwiseToQuantizeOps.back()->getOperand(0).getDefiningOp();
            }
        }();

        // Verify input operation meets fusion requirements
        if (auto result = verifyInput(input); result.failed()) {
            return result;
        }

        log.trace("[findQuantizeOrQuantizedNCE] Push Op {0} at {1}", input->getName(), input->getLoc());
        eltwiseToQuantizeOps.push_back(input);
    } while (!mlir::isa<IE::LayerWithPostOpInterface, IE::QuantizeOp>(eltwiseToQuantizeOps.back()));

    auto lastOp = eltwiseToQuantizeOps.back();

    const bool verifyLastOp = mlir::isa<IE::LayerWithPostOpInterface, IE::QuantizeOp>(lastOp);
    VPUX_THROW_UNLESS(verifyLastOp, "Expected lastOp to be NCE task or QuantizeOp");

    const std::string operationType = mlir::isa<IE::QuantizeOp>(lastOp) ? "Quantize" : "NCE task";
    log.trace("[findQuantizeOrQuantizedNCE] Found {0} {1} at {2}, stop pattern searching", operationType,
              lastOp->getName(), lastOp->getLoc());

    return mlir::success();
}

template <typename ConcreteOp>
mlir::LogicalResult IE::removeQuantOrFusedQuant(ConcreteOp origOp, mlir::PatternRewriter& rewriter,
                                                ArrayRef<mlir::Operation*> eltwiseToQuantizeOps,
                                                mlir::Operation* quantOrQuantizedNCE, mlir::Type elementType,
                                                Logger log) {
    if (mlir::isa_and_nonnull<IE::LayerWithPostOpInterface>(quantOrQuantizedNCE)) {
        const auto isPerChannel = IE::isPerAxisQuant(quantOrQuantizedNCE->getResult(0));
        auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(quantOrQuantizedNCE);
        if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(!isPerChannel)) {
            return matchFailed(rewriter, origOp, "Producer {0} is not supported", quantOrQuantizedNCE->getName());
        }

        auto* newNCETask = rewriter.clone(*quantOrQuantizedNCE);
        NDTypeInterface newType = newNCETask->getResult(0).getType();
        newType = newType.changeElemType(elementType);
        newNCETask->getResult(0).setType(newType);
        newNCETask->moveBefore(quantOrQuantizedNCE);

        log.trace("[removeQuantOrFusedQuant] Replace {0} {1} at {2} with {3} {4} at {5}",
                  quantOrQuantizedNCE->getName(), quantOrQuantizedNCE->getResult(0).getType(),
                  quantOrQuantizedNCE->getLoc(), newNCETask->getName(), newNCETask->getResult(0).getType(),
                  newNCETask->getLoc());

        rewriter.replaceOp(quantOrQuantizedNCE, newNCETask->getResult(0));
    } else if (mlir::isa_and_nonnull<IE::QuantizeOp>(quantOrQuantizedNCE)) {
        log.trace("[removeQuantOrFusedQuant] Remove {0} at {1}", quantOrQuantizedNCE->getName(),
                  quantOrQuantizedNCE->getLoc());
        rewriter.replaceOp(quantOrQuantizedNCE, quantOrQuantizedNCE->getOperand(0));
    } else {
        log.trace("[removeQuantOrFusedQuant] Quantize is already erased on lhs");
    }

    for (auto iterator = eltwiseToQuantizeOps.rbegin(); iterator != eltwiseToQuantizeOps.rend(); ++iterator) {
        log.trace("[removeQuantOrFusedQuant] Change {0} at {1} to {2}", (*iterator)->getName(), (*iterator)->getLoc(),
                  (*iterator)->getResult(0).getType());
        inferReturnTypes(*iterator, InferShapedTypeMode::ELEM_TYPE);
    }

    return mlir::success();
}

template <typename ConcreteOp>
mlir::LogicalResult QuantizeWithTwoInputsNCEEltwiseOpGeneric<ConcreteOp>::matchAndRewrite(
        ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    static_assert(ConcreteOp::template hasTrait<IE::EltwiseOp>(), "Expected operation to be EltwiseOp");
    VPUX_THROW_UNLESS(origOp.getNumOperands() == 2, "Expected operation to take two operands");

    auto isQuantizedInput = [](mlir::TypedValue<mlir::RankedTensorType> value) {
        return mlir::isa<mlir::quant::QuantizedType>(value.getType().getElementType());
    };
    const auto noQuantizedInput = !isQuantizedInput(origOp.getInput1()) && !isQuantizedInput(origOp.getInput2());
    if (noQuantizedInput) {
        return matchFailed(rewriter, origOp, "OrigOp doesn't have quantized input");
    }
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(origOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return matchFailed(rewriter, origOp, "OrigOp doesn't support mixed precision");
    }

    SmallVector<mlir::Operation*> lhsEltwiseToQuantizeOps, rhsEltwiseToQuantizeOps;
    // Walk through FakeQuantize-agnostic ops and find quantize or quantized NCE task
    if (auto result = findQuantizeOrQuantizedNCE<ConcreteOp>(origOp, rewriter, origOp.getInput1(),
                                                             lhsEltwiseToQuantizeOps, _log);
        result.failed()) {
        return result;
    }
    if (auto result = findQuantizeOrQuantizedNCE<ConcreteOp>(origOp, rewriter, origOp.getInput2(),
                                                             rhsEltwiseToQuantizeOps, _log);
        result.failed()) {
        return result;
    }
    _log.trace("[{0}] Pop {1} out of lhsEltwiseToQuantizeOps", this->getDebugName(),
               lhsEltwiseToQuantizeOps.back()->getName());
    mlir::Operation* lhsQuant = lhsEltwiseToQuantizeOps.pop_back_val();
    _log.trace("[{0}] Pop {1} out of rhsEltwiseToQuantizeOps", this->getDebugName(),
               rhsEltwiseToQuantizeOps.back()->getName());
    mlir::Operation* rhsQuant = rhsEltwiseToQuantizeOps.pop_back_val();

    if (mlir::isa<IE::LayerWithPostOpInterface>(lhsQuant) && mlir::isa<IE::LayerWithPostOpInterface>(rhsQuant)) {
        return matchFailed(rewriter, origOp, "Quantizes for both ancestors have been fused at {0} ({1}) and {2} ({3})",
                           lhsQuant->getName(), lhsQuant->getLoc(), rhsQuant->getName(), rhsQuant->getLoc());
    }
    // At this point at least one of lhsQuant and rhsQunt is QuantizeOp

    auto getInputElementType = [](mlir::Operation* operation) {
        return mlir::cast<NDTypeInterface>(operation->getOperand(0).getType()).getElementType();
    };
    const mlir::Type elementType =
            mlir::isa<IE::QuantizeOp>(lhsQuant) ? getInputElementType(lhsQuant) : getInputElementType(rhsQuant);

    // Detect if lhs and rhs use the same quantize to avoid double delete
    if (lhsQuant == rhsQuant) {
        rhsQuant = nullptr;
    }

    // Remove the quantize or fused-quantize, and update the type alone the way
    if (auto result = removeQuantOrFusedQuant<ConcreteOp>(origOp, rewriter, lhsEltwiseToQuantizeOps, lhsQuant,
                                                          elementType, _log);
        result.failed()) {
        return result;
    }
    if (auto result = removeQuantOrFusedQuant<ConcreteOp>(origOp, rewriter, rhsEltwiseToQuantizeOps, rhsQuant,
                                                          elementType, _log);
        result.failed()) {
        return result;
    }

    return mlir::success();
}

mlir::LogicalResult QuantizeWithAvgPool::matchAndRewrite(IE::AvgPoolOp avgPoolOp,
                                                         mlir::PatternRewriter& rewriter) const {
    const auto isInputQuantized = [&avgPoolOp]() {
        return mlir::isa<mlir::quant::QuantizedType>(avgPoolOp.getInput().getType().getElementType());
    }();
    if (!isInputQuantized) {
        return matchFailed(rewriter, avgPoolOp, "OrigOp doesn't have quantized input");
    }
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(avgPoolOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return matchFailed(rewriter, avgPoolOp, "OrigOp doesn't support mixed precision");
    }

    // Block transformation if AvgPool has a PPE
    if (::hasPPE(avgPoolOp.getOperation())) {
        return matchFailed(rewriter, avgPoolOp,
                           "AvgPool at {0} has fused PPE, skipping quantization fusion to preserve consistency",
                           avgPoolOp->getLoc());
    }

    SmallVector<mlir::Operation*> avgPoolToQuantizeOps;
    // Walk through FakeQuantize-agnostic ops and find quantize or quantized NCE task
    if (auto result = findQuantizeOrQuantizedNCE(avgPoolOp, rewriter, avgPoolOp.getInput(), avgPoolToQuantizeOps, _log);
        result.failed()) {
        return result;
    }

    _log.trace("[{0}] Pop {1} out of avgPoolToQuantizeOps", this->getDebugName(),
               avgPoolToQuantizeOps.back()->getName());
    mlir::Operation* quant = avgPoolToQuantizeOps.pop_back_val();

    if (!mlir::isa<IE::QuantizeOp>(quant)) {
        return matchFailed(rewriter, avgPoolOp, "Quantizes for ancestors have been fused at {0} ({1})",
                           quant->getName(), quant->getLoc());
    }

    auto quantizeOp = mlir::cast<IE::QuantizeOp>(quant);

    const mlir::Type elementType = mlir::cast<NDTypeInterface>(quantizeOp.getInput().getType()).getElementType();

    // Remove the quantize, and update the type alone the way
    if (auto result = removeQuantOrFusedQuant(avgPoolOp, rewriter, avgPoolToQuantizeOps, quant, elementType, _log);
        result.failed()) {
        return result;
    }

    return mlir::success();
}
template <typename ConcreteOp>
mlir::LogicalResult QuantizeWithNCEOp<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    const auto getInputVal = [&]() -> mlir::Value {
        if constexpr (std::is_same_v<ConcreteOp, IE::MatMulOp>) {
            return origOp.getInput1();
        } else {
            return origOp.getInput();
        }
    };

    const auto getWeightsVal = [&]() -> mlir::Value {
        if constexpr (std::is_same_v<ConcreteOp, IE::MatMulOp>) {
            return origOp.getInput2();
        } else {
            return origOp.getFilter();
        }
    };

    const auto isInput16BitsQuantized = [&]() -> bool {
        auto inputType = mlir::cast<vpux::NDTypeInterface>(getInputVal().getType());
        if (!inputType) {
            return false;
        }
        auto qType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputType.getElementType());
        if (qType != nullptr) {
            return qType.getStorageType().isInteger(16);
        }
        return false;
    }();

    if (!isInput16BitsQuantized) {
        return matchFailed(rewriter, origOp, "OrigOp doesn't have 16 bit integer quantized input");
    }
    if (!mlir::isa<Const::DeclareOp>(getWeightsVal().getDefiningOp())) {
        return matchFailed(rewriter, origOp, "Weights operand must be DeclareOp");
    }
    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(origOp.getOperation());
    if (!quantizedLayerOp || !quantizedLayerOp.isMixPrecisionSupported(false)) {
        return matchFailed(rewriter, origOp, "OrigOp doesn't support mixed precision");
    }

    SmallVector<mlir::Operation*> convToQuantizeOps;
    // Walk through FakeQuantize-agnostic ops and find quantize or quantized NCE task
    if (auto result = findQuantizeOrQuantizedNCE(origOp, rewriter, getInputVal(), convToQuantizeOps, _log);
        result.failed()) {
        return result;
    }

    _log.trace("[{0}] Pop {1} out of convToQuantizeOps", this->getDebugName(), convToQuantizeOps.back()->getName());
    mlir::Operation* quant = convToQuantizeOps.pop_back_val();

    if (!mlir::isa<IE::QuantizeOp>(quant)) {
        return matchFailed(rewriter, origOp, "Quantize for ancestors have been fused at {0} ({1})", quant->getName(),
                           quant->getLoc());
    }

    auto quantizeOp = mlir::cast<IE::QuantizeOp>(quant);

    const mlir::Type elementType = mlir::cast<NDTypeInterface>(quantizeOp.getInput().getType()).getElementType();

    // Remove the quantize, and update the type alone the way
    if (auto result = removeQuantOrFusedQuant(origOp, rewriter, convToQuantizeOps, quant, elementType, _log);
        result.failed()) {
        return result;
    }

    return mlir::success();
}
template class IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::AddOp>;
template class IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::MultiplyOp>;
template class IE::QuantizeWithTwoInputsNCEEltwiseOpGeneric<IE::SubtractOp>;

template class IE::QuantizeWithNCEOp<IE::ConvolutionOp>;
template class IE::QuantizeWithNCEOp<IE::MatMulOp>;
