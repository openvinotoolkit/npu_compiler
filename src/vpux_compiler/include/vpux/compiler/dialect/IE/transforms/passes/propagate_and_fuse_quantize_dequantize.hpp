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
                                  mlir::Type& origDstElemType, Logger log, ShapeRef scaleShape = {}) {
    auto elemTypeInfoOp = mlir::dyn_cast<IE::ElemTypeInfoOpInterface>(user);
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
    // For DynamicDequantize: verify the scale shape can be propagated through AffineReshape before modifying IR.
    if (!scaleShape.empty()) {
        if (auto affineReshapeOp = mlir::dyn_cast<IE::AffineReshapeOp>(user)) {
            const auto newScaleShape = computeShapeValueFromAffineReshape(affineReshapeOp, scaleShape);
            if (!newScaleShape.has_value()) {
                log.trace("Scale shape cannot be propagated through AffineReshape {0}", affineReshapeOp);
                return false;
            }
            // The reshape must preserve the total number of scale elements.  A mismatch means
            // a group dimension is being merged with a non-group dimension (e.g. the 16 groups
            // in [1,2048,16,1] folding into the output channel), which is semantically invalid
            // for per-group DynamicDequantize.
            const auto origElems = vpux::details::calcTotalShapeSize(scaleShape.raw());
            const auto newElems = vpux::details::calcTotalShapeSize(newScaleShape.value());
            if (origElems != newElems) {
                log.trace("Scale reshape through AffineReshape {0} changes element count ({1} -> {2}): "
                          "group dimensions cannot be merged",
                          affineReshapeOp, origElems, newElems);
                return false;
            }
        }
    }
    return true;
}

template <typename ConcreteOp>
class PropagateDequantize final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    PropagateDequantize(mlir::MLIRContext* ctx, Logger log, bool seOpsEnabled)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _log(log), _seOpsEnabled(seOpsEnabled) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
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
    _log.trace("PropagateDequantize Got layer: {0}", origOp);

    auto ctx = origOp->getContext();
    auto users = origOp->getUsers();
    mlir::Operation* lastUser = nullptr;
    auto origDstElemType = origOp.getDstElemType();
    auto quantizedElemType = mlir::cast<vpux::NDTypeInterface>(origOp.getInput().getType()).getElementType();
    // In case of Dynamic DequantizeOp we only support to propagate the quantization only in case we have 1 dim != 1 in
    // the scale table. This ensures us we do not have groups in the convolution.
    mlir::Value scaleInput = nullptr;
    if (auto dynamicDequantizeOp = mlir::dyn_cast<IE::DynamicDequantizeOp>(origOp.getOperation())) {
        scaleInput = dynamicDequantizeOp.getScale();
        if (scaleInput != nullptr) {
            auto numOfNonOneDims = getNonOneDim(getShape(scaleInput));
            auto hasOneUseDynamicDequantizeOp = dynamicDequantizeOp->hasOneUse();
            auto childOp = *(dynamicDequantizeOp->getUsers().begin());
            if ((numOfNonOneDims.size() != 1 || !hasOneUseDynamicDequantizeOp) &&
                mlir::isa_and_present<IE::TransposeOp>(childOp)) {
                _log.trace("Only support unrolled DynamicDequantizeOp, but got {0}", numOfNonOneDims);
                return mlir::failure();
            }
        }
        if (dynamicDequantizeOp.getZp() != nullptr) {
            _log.trace("Only support unrolled DynamicDequantizeOp without zero point, but got one");
            return mlir::failure();
        }
    }
    // store orig users for dequantize-like ops to avoid update missing and duplicate same users
    // SetVector preserves insertion order while keeping uniqueness
    const ShapeRef currentScaleShape = scaleInput != nullptr ? getShape(scaleInput) : ShapeRef{};
    llvm::SetVector<mlir::Operation*> origUsers;
    if (origOp->hasOneUse()) {
        // DequantizeOp/DynamicDequantizeOp -> Op1 -> Op2 -> Op3, find Op3
        mlir::Operation* nextUser = *(users.begin());
        while (nextUser) {
            if (!isValidToPropagateDequantize<ConcreteOp>(nextUser, _seOpsEnabled, quantizedElemType, origDstElemType,
                                                          _log, currentScaleShape)) {
                break;
            }
            if (nextUser == *(users.begin())) {
                origUsers.insert(nextUser);
            }
            lastUser = nextUser;
            // Not forward for multiple users
            if (!nextUser->hasOneUse()) {
                break;
            }
            nextUser = *(nextUser->getUsers().begin());
        }
    } else {
        for (auto user : users) {
            if (isValidToPropagateDequantize<ConcreteOp>(user, _seOpsEnabled, quantizedElemType, origDstElemType, _log,
                                                         currentScaleShape)) {
                origUsers.insert(user);
            }
        }
    }
    if (origUsers.empty()) {
        return mlir::failure();
    }

    // 4. Rewrite the sub-graph.
    for (auto user : origUsers) {
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

            if (mlir::isa_and_present<IE::DynamicDequantizeOp>(origOp)) {
                mlir::IRMapping mapper;
                auto scaleCurrentUser = rewriter.clone(*currentUser, mapper);
                if (mlir::isa<IE::AffineReshapeOp>(scaleCurrentUser)) {
                    auto newShapeValue = computeShapeValueFromAffineReshape(
                            mlir::cast<IE::AffineReshapeOp>(scaleCurrentUser), getShape(scaleInput));
                    // newShapeValue is guaranteed valid: isValidToPropagateDequantize pre-checked this.
                    scaleCurrentUser->setAttr("shape_value", getIntArrayAttr(ctx, newShapeValue.value()));
                }
                scaleCurrentUser->setOperand(0, scaleInput);
                vpux::inferReturnTypes(scaleCurrentUser, vpux::InferShapedTypeMode::ALL);
                scaleInput = scaleCurrentUser->getResult(0);
            }

            for (auto [outputInd, output] : llvm::enumerate(currentUser->getResults())) {
                output.setType(inferredTypes[outputInd]);
                if (isToInsert) {
                    rewriter.setInsertionPointAfter(currentUser);
                    auto newLoc = appendLoc(currentUser->getLoc(), "propagated_Dequantize '{0}'", outputInd);

                    mlir::IRMapping mapper;
                    auto newDequant = rewriter.clone(*origOp, mapper);
                    newDequant->setOperand(0, output);
                    if (mlir::isa_and_present<IE::DynamicDequantizeOp>(origOp)) {
                        newDequant->setOperand(1, scaleInput);
                    }
                    newDequant->setLoc(newLoc);
                    vpux::inferReturnTypes(newDequant, vpux::InferShapedTypeMode::ALL);

                    _log.trace("Added new Dequantize op: '{0}' at index '{1}'", newDequant, outputInd);
                    output.replaceAllUsesExcept(newDequant->getResult(0),
                                                llvm::SmallPtrSet<mlir::Operation*, 1>{newDequant});
                    _log.trace("All uses of current layer have been replaced with new Dequantize op at index '{0}'",
                               outputInd);
                }
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

}  // namespace vpux::IE
