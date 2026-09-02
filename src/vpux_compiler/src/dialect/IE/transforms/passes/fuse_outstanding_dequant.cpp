//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/attributes.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <mlir/IR/OperationSupport.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSEOUTSTANDINGDEQUANT
#define GEN_PASS_DEF_FUSEOUTSTANDINGDEQUANT
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

// Helper function to check if operation is allowed to be walked through during backward search from Dequantize.
// These are view-like/layout transformation operations that reorganize data without computation.
// They're transparent to quantization - they don't change numerical values or quantization parameters,
// only how the data is organized in memory.
bool isAllowedElemTypeOp(mlir::Operation* op) {
    return mlir::isa<IE::AffineReshapeOp, IE::ExpandOp, IE::ExpandDilatedOp, IE::ReorderOp, IE::ReshapeOp,
                     IE::TransposeOp, IE::SliceOp>(op);
}

struct NCEMatchResult {
    mlir::Operation* nceOp = nullptr;
    SmallVector<mlir::Operation*> chain;
};

mlir::FailureOr<NCEMatchResult> findNCEProducer(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter, Logger log,
                                                StringRef debugName) {
    auto* producer = origOp.getInput().getDefiningOp();
    if (producer == nullptr) {
        return matchFailed(log, rewriter, origOp, "[{0}] Producer is a block argument", debugName);
    }
    if (!producer->getResult(0).hasOneUse()) {
        return matchFailed(log, rewriter, origOp, "[{0}] Producer has more than one consumer", debugName);
    }

    NCEMatchResult result;
    if (mlir::isa<IE::QuantizedLayerOpInterface>(producer)) {
        result.nceOp = producer;
        return result;
    }

    log.trace("[{0}] Search quantized NCE task for {1} at {2}", debugName, origOp->getName(), origOp->getLoc());
    mlir::Operation* operation = origOp;
    while (operation != nullptr) {
        auto* input = operation->getOperand(0).getDefiningOp();
        if (input == nullptr) {
            return matchFailed(log, rewriter, origOp, "[{0}] Reached block argument while searching for NCE producer",
                               debugName);
        }

        const bool isAllowedElemOp = isAllowedElemTypeOp(input);
        const bool isNCE = mlir::isa<IE::QuantizedLayerOpInterface>(input);
        if (!isAllowedElemOp && !isNCE) {
            return matchFailed(log, rewriter, origOp,
                               "[{0}] Ancestor {1} at {2} is neither allowed operation nor NCE operation", debugName,
                               input->getName(), input->getLoc());
        }
        if (!input->hasOneUse()) {
            return matchFailed(log, rewriter, origOp, "[{0}] Ancestor {1} at {2} has more than one consumer", debugName,
                               input->getName(), input->getLoc());
        }

        if (isAllowedElemOp) {
            if (input->getNumOperands() > 1) {
                return matchFailed(log, rewriter, origOp, "[{0}] Ancestor {1} at {2} has more than one ancestors",
                                   debugName, input->getName(), input->getLoc());
            }
            log.trace("[{0}] Push allowed operation {1} at {2}", debugName, input->getName(), input->getLoc());
            result.chain.push_back(input);
            operation = input;
            continue;
        }

        log.trace("[{0}] Found quantized layer {1} at {2}, stop pattern searching", debugName, input->getName(),
                  input->getLoc());
        result.nceOp = input;
        return result;
    }

    return matchFailed(log, rewriter, origOp, "[{0}] NCE producer not found", debugName);
}

// Legality check:
//  1. If post-op doesn't exist it's valid.
//  2. If post-op exists it must be ReLU.
mlir::LogicalResult validateLayer(IE::LayerWithPostOpInterface layerWithPostOp, Logger log, StringRef debugName,
                                  mlir::PatternRewriter& rewriter, IE::DequantizeOp origOp) {
    // Op that doesn't support post-op is also valid
    if (layerWithPostOp == nullptr) {
        return mlir::success();
    }

    // Bail on non-ReLU post_op (cannot combine with saturation compensation)
    if (auto postOpAttr = layerWithPostOp.getPostOp()) {
        if (!mlir::isa<IE::ReluAttr>(postOpAttr)) {
            return matchFailed(log, rewriter, origOp,
                               "[{0}] {1} has non-ReLU postOp ({2}), preserving Dequantize at loc '{3}'", debugName,
                               layerWithPostOp->getName().getStringRef(), postOpAttr.getName(), origOp->getLoc());
        }
    }

    return mlir::success();
}

struct Clamp {
    float min;
    float max;
};

struct MergeClamp {
    Clamp newClamp;
    std::optional<Clamp> existingClamp;
};

MergeClamp getMergedClampValues(IE::LayerWithPostOpInterface layerWithPostOp, float satMin, float satMax) {
    // Check for clamp attribute, collect existing clamp values and new clamp values if attribute clamp and parameter
    // clamps were to be merged.
    // NOTE: There exist hardware differences for clamp attributes. It's easier to avoid
    //       thinking about clamp attributes instead think of clamps as explicit.
    //       Let SwapViewLikeOpAndClamp and FuseActivationOps handle fusing clamps after this pass.
    MergeClamp mergeClamp = {};
    if (layerWithPostOp != nullptr) {
        if (auto existingClamp = layerWithPostOp.getClampAttr()) {
            auto existingMin =
                    static_cast<float>(mlir::cast<mlir::FloatAttr>(existingClamp.get("min")).getValueAsDouble());
            auto existingMax =
                    static_cast<float>(mlir::cast<mlir::FloatAttr>(existingClamp.get("max")).getValueAsDouble());

            satMin = std::max(satMin, existingMin);
            satMax = std::min(satMax, existingMax);

            mergeClamp.existingClamp = {existingMin, existingMax};
        }
    }

    mergeClamp.newClamp = {satMin, satMax};

    return mergeClamp;
}

bool isClampDisjointed(Clamp newClamp, std::optional<Clamp> existingClamp, Clamp dequantizeClamp,
                       mlir::Location dequantizeLoc, Logger log, StringRef debugName) {
    if (newClamp.min <= newClamp.max) {
        return false;
    }

    // NOTE: Disjoint clamps indicate an issue with clamps from DequantizeOp or layerWithPostOp
    log.warning("[{0}] Disjoint clamp calculated with range [{1}, {2}] (clamp's min is larger than its max).",
                debugName, newClamp.min, newClamp.max);
    if (existingClamp.has_value()) {
        log.warning("[{0}] This might indicate a problem with the clamp attribute is [{1}, {2}]", debugName,
                    existingClamp.value().min, existingClamp.value().max);
    } else {
        log.warning("[{0}] This might indicate a problem with the clamp from Dequantize at loc '{1}', its range is "
                    "[{2}, {3}]",
                    debugName, dequantizeLoc, dequantizeClamp.min, dequantizeClamp.max);
    }

    return true;
}

}  // namespace

//
// FuseOutstandingDequantPass
//

class FuseOutstandingDequantPass final : public IE::impl::FuseOutstandingDequantBase<FuseOutstandingDequantPass> {
public:
    explicit FuseOutstandingDequantPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;
};

mlir::LogicalResult FuseOutstandingDequantPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    return mlir::success();
}

class DequantizeWithNCEReLUxRewriter final : public mlir::OpRewritePattern<IE::DequantizeOp> {
public:
    DequantizeWithNCEReLUxRewriter(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::DequantizeOp>(ctx), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult DequantizeWithNCEReLUxRewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    const auto dequantInputType = origOp.getInput().getType();
    const auto elemType = mlir::cast<mlir::ShapedType>(dequantInputType).getElementType();
    const auto dequantUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    const bool isPerChannel = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    const auto dequantOutputType = origOp.getOutput().getType();

    auto matchResult = findNCEProducer(origOp, rewriter, _log, this->getDebugName());
    if (mlir::failed(matchResult)) {
        return mlir::failure();
    }
    auto* nceOp = matchResult->nceOp;
    auto& chain = matchResult->chain;

    if (dequantUniformType == nullptr) {
        _log.trace("[{0}] Dequantize is per-axes", this->getDebugName());
    }

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(nceOp);
    if (quantizedLayerOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "[{0}] Producer is not a quantized layer operation",
                           this->getDebugName());
    }

    if (!quantizedLayerOp.isMixPrecisionSupported(!isPerChannel)) {
        return matchFailed(_log, rewriter, origOp, "[{0}] Producer {1} is not supported", this->getDebugName(),
                           nceOp->getName());
    }

    // Must check nceOp because if invalid we return mlir::failure() and it's illegal to modify IR and return failure
    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(nceOp);
    if (mlir::failed(validateLayer(layerWithPostOp, _log, this->getDebugName(), rewriter, origOp))) {
        return mlir::failure();
    }

    std::optional<Clamp> reluxClamp;
    if (dequantUniformType != nullptr) {
        int64_t levels = 0;
        Clamp dequantClamp = {};
        getFakeQuantParams(dequantUniformType, levels, dequantClamp.min, dequantClamp.max);
        auto [newClamp, existingClamp] = getMergedClampValues(layerWithPostOp, dequantClamp.min, dequantClamp.max);
        if (isClampDisjointed(newClamp, existingClamp, dequantClamp, origOp->getLoc(), _log, this->getDebugName())) {
            return mlir::failure();
        }
        if (existingClamp.has_value()) {
            const auto logCb = [&](const formatv_object_base& msg) {
                _log.trace("[{0}] {1}", this->getDebugName(), msg.str());
            };

            // NOTE: Later we might not create ReLUx, but we will still fuse Dequantize making this attribute
            //       no longer supported by HW. This would cause accuracy issue if not checked.
            if (!layerWithPostOp.isSupportedClampProperties(existingClamp.value().min, existingClamp.value().max,
                                                            dequantOutputType, logCb)) {
                return mlir::failure();
            }
        }

        // NOTE: This is ReLU(x)/Clamp(0, x) because hardware supports it, and would be more accurate than ReLU
        if (!isPerChannel && isFloatEqual(newClamp.min, 0.0f)) {
            reluxClamp = newClamp;
        } else {
            _log.trace("[{0}] Clamp range [{1}, {2}] not supported", this->getDebugName(), newClamp.min, newClamp.max);
        }
    }

    mlir::Operation* newNCEOp = rewriter.clone(*nceOp);
    if (chain.empty()) {
        newNCEOp->getResult(0).setType(dequantOutputType);
    } else {
        vpux::NDTypeInterface newType = newNCEOp->getResult(0).getType();
        newType = newType.changeElemType(dequantOutputType.getElementType());
        newNCEOp->getResult(0).setType(newType);
        newNCEOp->moveAfter(nceOp);
    }

    mlir::Value chainTail = chain.empty() ? newNCEOp->getResult(0) : chain.front()->getResult(0);
    auto result = chainTail;

    if (reluxClamp.has_value()) {
        const auto newClamp = reluxClamp.value();
        if (auto validLayerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(newNCEOp)) {
            validLayerWithPostOp.setClampAttr(nullptr);
        }

        rewriter.setInsertionPointAfter(newNCEOp);
        auto clampOp = rewriter.create<IE::ClampOp>(takeOpLoc(origOp, "as_clamp"), newNCEOp->getResult(0),
                                                    rewriter.getF64FloatAttr(newClamp.min),
                                                    rewriter.getF64FloatAttr(newClamp.max));
        // NOTE: The chain references based of nceOp not newNCEOp
        rewriter.replaceAllUsesExcept(nceOp->getResult(0), clampOp.getOutput(), clampOp);
        result = chain.empty() ? clampOp.getOutput() : chainTail;

        _log.trace("[{0}] Clamp range [{1}, {2}] is supported. Created Clamp at loc '{3}'", this->getDebugName(),
                   newClamp.min, newClamp.max, clampOp.getLoc());
    }

    _log.trace("[{0}] Replace {1} {2} at {3} with {4} {5} at {6}", this->getDebugName(), nceOp->getName(),
               nceOp->getResult(0).getType(), nceOp->getLoc(), newNCEOp->getName(), newNCEOp->getResult(0).getType(),
               newNCEOp->getLoc());
    rewriter.replaceOp(nceOp, newNCEOp->getResult(0));

    for (auto* viewOp : llvm::reverse(chain)) {
        _log.trace("[{0}] Change {1} at {2} to {3}", this->getDebugName(), viewOp->getName(), viewOp->getLoc(),
                   viewOp->getResult(0).getType());
        inferReturnTypes(viewOp, InferShapedTypeMode::ELEM_TYPE);
    }

    rewriter.replaceOp(origOp, result);

    return mlir::success();
}

void FuseOutstandingDequantPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<DequantizeWithNCEReLUxRewriter>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));
}

//
// createFuseOutstandingDequant
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseOutstandingDequant(Logger log) {
    return std::make_unique<FuseOutstandingDequantPass>(log);
}
