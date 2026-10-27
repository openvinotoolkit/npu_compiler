//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/attributes/shape.hpp"
#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/utils/elem_type_info_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/reshape_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
#include <mlir/Dialect/Quant/IR/QuantTypes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

namespace vpux::IE {

//
// FuseDequantizeWithMultiplier
//

class FuseDequantizeWithMultiplier final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    FuseDequantizeWithMultiplier(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
        setDebugName("FuseDequantizeWithMultiplier");
    }

    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

//
// PropagateQuantize
//

class PropagateQuantize final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    PropagateQuantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log), _seOpsEnabled(seOpsEnabled) {
    }

    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
    bool _seOpsEnabled;
};

//
// PropagateDequantize
//

template <typename ConcreteOp>
bool isValidToPropagateDequantize(mlir::Operation* user, bool seOpsEnabled, mlir::Type& quantizedElemType,
                                  mlir::Type& origDstElemType, Logger log) {
    auto elemTypeInfoOp = mlir::dyn_cast_if_present<IE::ElemTypeInfoOpInterface>(user);
    if (!elemTypeInfoOp) {
        return false;
    }

    const auto logCb = [&](const formatv_object_base& msg) {
        log.trace("{0}", msg.str());
    };
    const auto isSameDequantize = [&](mlir::Value input) {
        if (auto currentDequantize = input.getDefiningOp<ConcreteOp>()) {
            return currentDequantize.getDstElemType() == origDstElemType;
        }

        return false;
    };
    auto layer = mlir::cast<IE::LayerOpInterface>(user);
    auto layerOperandsSize = layer->getOperands().size();
    // 1. All inputs are dequantize-like ops of the same concrete type with the same destination element type
    if (layerOperandsSize > 1 && !llvm::all_of(layer.getInputs(), isSameDequantize)) {
        log.trace("The inputs of Operation {0} should all be the same dequantizeOp when Op operands > 1",
                  elemTypeInfoOp);
        return false;
    }
    // 2. Check if operation supports quantization params propagation.
    if (auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(user);
        layerWithPostOp != nullptr && layerWithPostOp.hasPPE()) {
        // A quantization-agnostic operation is no longer quantization-agnostic after it is fused with a post-op
        // (because post-op's are not quantization-agnostic). Since most post-op's will be fused by this time, this
        // check is here to prevent the propagation of input quantization through both the ElemTypeInfoOp and its
        // post-op. (At this time MaxPool seems to be the only operation which is both a IE::ElemTypeInfoOpInterface
        // and a IE::LayerWithPostOpInterface)
        log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
        return false;
    }
    // 3. Particular check for SE pointers
    if (!vpux::IE::isSupportedElemTypeInfoCase(elemTypeInfoOp.getOperation(), seOpsEnabled, logCb)) {
        log.trace("Operation {0} does not support SE pointers", elemTypeInfoOp);
        return false;
    }
    // 4. Check whether elemTypeInfoOp all input dequantizeOps parameters are consistent
    auto elemTypeInfo = elemTypeInfoOp.getElemTypeInfo();

    SmallVector<mlir::Type> originalTypes;
    originalTypes.reserve(layerOperandsSize);
    for (auto [idx, input] : llvm::enumerate(layer.getInputs())) {
        if (layerOperandsSize > 1) {
            auto dequantizeOp = input.getDefiningOp();
            quantizedElemType =
                    mlir::cast<vpux::NDTypeInterface>(dequantizeOp->getOperand(0).getType()).getElementType();
        }
        elemTypeInfo.setInput(idx, quantizedElemType);
        originalTypes.push_back(quantizedElemType);
    }
    elemTypeInfoOp.inferElemTypeInfo(elemTypeInfo);
    const auto typesAreOriginal = llvm::all_of(irange(originalTypes.size()), [&](size_t idx) {
        return elemTypeInfo.getInput(idx) == originalTypes[idx];
    });

    if (!typesAreOriginal) {
        log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
        return false;
    }
    // 5. Check whether elemTypeInfoOp all output parameters are consistent
    for (size_t outputInd = 0; outputInd < layer->getNumResults(); outputInd++) {
        if (!mlir::isa<mlir::quant::QuantizedType>(elemTypeInfo.getOutput(outputInd))) {
            log.trace("Operation {0} does not support quantization params propagation", elemTypeInfoOp);
            return false;
        }
    }
    if (layerOperandsSize == 1) {
        quantizedElemType = elemTypeInfo.getOutput(0);
        origDstElemType = elemTypeInfoOp.getElemTypeInfo().getOutput(0);
    }

    return true;
}

// Per-match state used to track how a ConcreteOp's non-activation (secondary) input, e.g. a
// DynamicDequantize's scale operand, evolves hop-by-hop as it is propagated through a chain of
// activation ops. It is declared once per matchAndRewrite call and threaded through the virtual
// hooks below by reference, instead of being kept as mutable pattern state. Unused by patterns
// without a secondary input (e.g. plain Dequantize).
struct NonActivationState {
    Shape secondaryShape{};
    mlir::Value secondaryValue = nullptr;
};

template <typename ConcreteOp>
class PropagateDequantize : public mlir::OpRewritePattern<ConcreteOp> {
public:
    PropagateDequantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _seOpsEnabled(seOpsEnabled) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual void mapNonActivationInput(ConcreteOp /*op*/, mlir::Operation* /*crtUser*/, mlir::IRMapping& /*mapper*/,
                                       mlir::PatternRewriter& /*rewriter*/, NonActivationState& /*state*/) const {
        // Op doesn't have a secondary input, nothing to do.
    }

    virtual mlir::LogicalResult validateOpParams(ConcreteOp /*op*/, mlir::PatternRewriter& /*rewriter*/,
                                                 NonActivationState& /*state*/) const {
        // nothing to check, op is always valid
        return mlir::success();
    }

    virtual bool isValidToPropagateDequant(ConcreteOp origOp, mlir::Operation* nextUser,
                                           NonActivationState& /*state*/) const {
        auto origDstElemType = origOp.getDstElemType();
        auto quantizedElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
        return isValidToPropagateDequantize<ConcreteOp>(nextUser, _seOpsEnabled, quantizedElemType, origDstElemType,
                                                        _log);
    }

    Logger _log;
    bool _seOpsEnabled;
};

/* This rewriter searches for pattern:
quantized_tensor -> [Dequantize/DynamicDequantize] -> fp_tensor -> [ElemTypeInfoOpInterface]                  ->
fp_tensor and replaces it with quantized_tensor -> [ElemTypeInfoOpInterface] -> quantized_tensor(inferred) ->
[Dequantize/DynamicDequantize] -> fp_tensor */
template <typename ConcreteOp>
mlir::LogicalResult PropagateDequantize<ConcreteOp>::matchAndRewrite(ConcreteOp origOp,
                                                                     mlir::PatternRewriter& rewriter) const {
    _log.trace("Got layer '{0}' @ {1}", origOp->getName(), origOp->getLoc());

    auto ctx = origOp->getContext();
    auto users = origOp->getUsers();
    mlir::Operation* lastUser = nullptr;
    // Scale-path state right after validateOpParams, used as the starting point for every
    // independent propagation chain below.
    NonActivationState baseState{};
    // In case of Dynamic DequantizeOp we only support to propagate the quantization only in case we have 1 dim != 1
    // in the scale table. This ensures us we do not have groups in the convolution.
    if (mlir::failed(validateOpParams(origOp, rewriter, baseState))) {
        return mlir::failure();
    }

    // store orig users for dequantize-like ops to avoid update missing and duplicate same users
    // SetVector preserves insertion order while keeping uniqueness
    llvm::SetVector<mlir::Operation*> origUsers;
    // Each entry in origUsers gets its own scale-path state, started fresh from baseState. This
    // matters when origOp has multiple direct users: each is an independent propagation chain and
    // must not observe state mutated while validating/rewriting a sibling chain.
    llvm::DenseMap<mlir::Operation*, NonActivationState> perUserState;
    if (origOp->hasOneUse()) {
        // DequantizeOp/DynamicDequantizeOp -> Op1 -> Op2 -> Op3, find Op3
        auto* const firstUser = *(users.begin());
        mlir::Operation* nextUser = firstUser;
        NonActivationState state = std::move(baseState);
        while (nextUser) {
            if (!isValidToPropagateDequant(origOp, nextUser, state)) {
                break;
            }
            lastUser = nextUser;
            // Not forward for multiple users
            if (!nextUser->hasOneUse()) {
                break;
            }
            nextUser = *(nextUser->getUsers().begin());
        }
        if (lastUser != nullptr) {
            origUsers.insert(firstUser);
            perUserState[firstUser] = std::move(state);
        }
    } else {
        for (auto user : users) {
            NonActivationState state = baseState;
            if (isValidToPropagateDequant(origOp, user, state)) {
                origUsers.insert(user);
                perUserState[user] = std::move(state);
            }
        }
    }
    if (origUsers.empty()) {
        return mlir::failure();
    }

    // 4. Rewrite the sub-graph.
    for (auto user : origUsers) {
        auto& state = perUserState[user];
        rewriter.setInsertionPointAfter(origOp);
        rewriter.startOpModification(user);
        // remove all dequantize-like ops op operands
        const auto inputs = user->getOpOperands();
        auto layer = mlir::cast<IE::LayerOpInterface>(user);
        for (auto idx : irange(inputs.size())) {
            auto& input = inputs[idx];
            auto dequantizeOp = layer.getInputs()[idx].getDefiningOp<ConcreteOp>();
            input.set(dequantizeOp.getInput());
        }
        // infer return type and insert new Dequantize/DynamicDequantize Op
        auto currentUser = user;
        while (currentUser) {
            SmallVector<mlir::Type> inferredTypes;
            inferredTypes.reserve(currentUser->getNumResults());
            auto op = mlir::cast<mlir::InferTypeOpInterface>(currentUser);
            VPUX_THROW_UNLESS(op.inferReturnTypes(ctx, op->getLoc(), op->getOperands(), op->getAttrDictionary(),
                                                  op->getPropertiesStorage(), op->getRegions(), inferredTypes)
                                      .succeeded(),
                              "New type inference failed for '{0}'", op);

            auto isToInsert = !lastUser || currentUser == lastUser;

            mlir::IRMapping mapper;
            mapNonActivationInput(origOp, currentUser, mapper, rewriter, state);

            for (auto [outputInd, output] : llvm::enumerate(currentUser->getResults())) {
                output.setType(inferredTypes[outputInd]);
                if (!isToInsert) {
                    continue;
                }

                rewriter.setInsertionPointAfter(currentUser);
                auto newLoc = appendLoc(currentUser->getLoc(), "propagated_dequant_{0}", outputInd);

                // Setting activation path input to be the output of the previous op, by remapping it.
                // Don't clear mapper, since it holds the mappings for the secondary input.
                mapper.map(origOp.getInput(), output);
                auto newDequant = rewriter.clone(*origOp, mapper);
                newDequant->setLoc(newLoc);
                vpux::inferReturnTypes(newDequant, vpux::InferShapedTypeMode::ALL);

                _log.trace("Added new '{0}' op: '{1}' at index '{2}'", newDequant->getName(), newDequant, outputInd);
                output.replaceAllUsesExcept(newDequant->getResult(0),
                                            llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});
                _log.trace("All uses of current layer have been replaced with new Dequantize op at index '{0}'",
                           outputInd);
            }
            if (isToInsert) {
                break;
            }
            currentUser = *(currentUser->getUsers().begin());
        }
        rewriter.finalizeOpModification(user);
    }

    return mlir::success();
}

class PropagateDynamicDequantize final : public PropagateDequantize<IE::DynamicDequantizeOp> {
public:
    PropagateDynamicDequantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : PropagateDequantize<IE::DynamicDequantizeOp>(ctx, log, seOpsEnabled) {
    }

protected:
    void mapNonActivationInput(IE::DynamicDequantizeOp op, mlir::Operation* crtUser, mlir::IRMapping& map,
                               mlir::PatternRewriter& rewriter, NonActivationState& state) const override;
    mlir::LogicalResult validateOpParams(IE::DynamicDequantizeOp op, mlir::PatternRewriter& rewriter,
                                         NonActivationState& state) const override;
    bool isValidToPropagateDequant(IE::DynamicDequantizeOp op, mlir::Operation* nextUser,
                                   NonActivationState& state) const override;
};

}  // namespace vpux::IE
