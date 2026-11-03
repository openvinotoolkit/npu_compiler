//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_outstanding_dequant.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/activation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <algorithm>
#include <optional>

using namespace vpux;
using namespace IE;

namespace {

// Helper function to check if operation is allowed to be walked through during backward search from Dequantize.
// These are view-like/layout transformation operations that reorganize data without computation.
// They're transparent to quantization - they don't change numerical values or quantization parameters,
// only how the data is organized in memory.
bool isAllowedElemTypeOp(mlir::Operation* op) {
    return mlir::isa_and_nonnull<IE::AffineReshapeOp, IE::ExpandOp, IE::ExpandDilatedOp, IE::ReorderOp, IE::ReshapeOp,
                                 IE::TransposeOp, IE::SliceOp, IE::ShapeCastOp, IE::PermuteCastOp, IE::LayoutCastOp,
                                 IE::MemPermuteOp>(op);
}

struct NCEMatchResult {
    mlir::Operation* nceOp = nullptr;
    SmallVector<mlir::Operation*> chain;
};

mlir::FailureOr<NCEMatchResult> findNCEProducer(IE::DequantizeOp origOp, mlir::PatternRewriter& rewriter, Logger log) {
    auto* producer = origOp.getInput().getDefiningOp();
    if (producer == nullptr) {
        return matchFailed(log, rewriter, origOp, "Producer is a block argument");
    }
    if (!producer->getResult(0).hasOneUse()) {
        return matchFailed(log, rewriter, origOp, "Producer has more than one consumer");
    }

    NCEMatchResult result;
    if (mlir::isa<IE::QuantizedLayerOpInterface>(producer)) {
        result.nceOp = producer;
        return result;
    }

    log.trace("Search quantized NCE task for Op '{0}' at loc '{1}'", origOp->getName(), origOp->getLoc());
    mlir::Operation* operation = origOp;
    while (auto* input = operation->getOperand(0).getDefiningOp()) {
        const bool isAllowedElemOp = isAllowedElemTypeOp(input);
        const bool isNCE = mlir::isa<IE::QuantizedLayerOpInterface>(input);
        if (!isAllowedElemOp && !isNCE) {
            return matchFailed(log, rewriter, origOp,
                               "Op '{0}' at loc '{1}' is neither allowed operation nor NCE operation", input->getName(),
                               input->getLoc());
        }
        if (!input->hasOneUse()) {
            return matchFailed(log, rewriter, origOp, "Op '{0}' at loc '{1}' has more than one consumer",
                               input->getName(), input->getLoc());
        }

        if (!isAllowedElemOp) {
            log.trace("Found quantized layer '{0}' at loc '{1}', stop pattern searching", input->getName(),
                      input->getLoc());
            result.nceOp = input;
            return result;
        }

        if (input->getNumOperands() > 1) {
            return matchFailed(log, rewriter, origOp, "Op '{0}' at loc '{1}' has more than one ancestors",
                               input->getName(), input->getLoc());
        }

        log.trace("Push allowed operation '{0}' at loc '{1}'", input->getName(), input->getLoc());
        result.chain.push_back(input);
        operation = input;
    }

    return matchFailed(log, rewriter, origOp, "NCE producer not found");
}

struct Clamp {
    float min;
    float max;
};

struct GatheredClamps {
    std::optional<Clamp> existingClamp;
    Clamp intersectedClamp;
};

GatheredClamps gatherClamps(IE::LayerWithPostOpInterface layerWithPostOp, float dequantizeMin, float dequantizeMax) {
    // Check for clamp attribute, collect existing clamp values and new clamp values if attribute clamp and parameter
    // clamps were to be merged.
    GatheredClamps gatheredClamps = {};
    if (auto existingClamp = layerWithPostOp.getClampAttr()) {
        auto existingMin = static_cast<float>(mlir::cast<mlir::FloatAttr>(existingClamp.get("min")).getValueAsDouble());
        auto existingMax = static_cast<float>(mlir::cast<mlir::FloatAttr>(existingClamp.get("max")).getValueAsDouble());

        dequantizeMin = std::max(dequantizeMin, existingMin);
        dequantizeMax = std::min(dequantizeMax, existingMax);

        gatheredClamps.existingClamp = {existingMin, existingMax};
    }

    gatheredClamps.intersectedClamp = {dequantizeMin, dequantizeMax};

    return gatheredClamps;
}

mlir::Operation* cloneAndRetypeNCEOp(mlir::Operation* nceOp, mlir::ShapedType dequantOutputType,
                                     mlir::PatternRewriter& rewriter) {
    mlir::Operation* newNCEOp = rewriter.clone(*nceOp);
    vpux::NDTypeInterface newType = newNCEOp->getResult(0).getType();
    newType = newType.changeElemType(dequantOutputType.getElementType());
    rewriter.modifyOpInPlace(newNCEOp, [&]() {
        newNCEOp->getResult(0).setType(newType);
        newNCEOp->moveAfter(nceOp);
    });

    return newNCEOp;
}

mlir::Value computeChainTail(ArrayRef<mlir::Operation*> chain, mlir::Operation* newNCEOp) {
    return chain.empty() ? newNCEOp->getResult(0) : chain.front()->getResult(0);
}

bool isNonReluPostOpSupported(IE::DequantizeOp origOp, IE::QuantizedLayerOpInterface quantizedLayerOp) {
    const auto inputType = mlir::cast<mlir::ShapedType>(origOp.getInput().getType());
    const auto quantizedType = mlir::dyn_cast<mlir::quant::QuantizedType>(inputType.getElementType());
    const bool is16BitQuantized = quantizedType != nullptr && quantizedType.getStorageType().isInteger(16);
    return config::getArch(quantizedLayerOp.getOperation()) > config::ArchKind::NPU50XX && is16BitQuantized;
}

// Validate the newNCEOp is legal, replace nceOp with cloned newNCEOp which has output type of Dequantize,
// and re-infer elem types through view-like chain between NCEOp and Dequantize
mlir::LogicalResult finalizeFusion(IE::DequantizeOp origOp, mlir::Operation* nceOp, mlir::Operation* newNCEOp,
                                   IE::QuantizedLayerOpInterface quantizedLayerOp, ArrayRef<mlir::Operation*> chain,
                                   mlir::PatternRewriter& rewriter, Logger log) {
    if (!IE::checkRescaledQuantApproximationForConvBasedOp(newNCEOp)) {
        log.trace("Mixed-precision rescale is not representable for Op '{0}' at loc '{1}', preserving "
                  "Dequantize at loc '{2}'",
                  newNCEOp->getName(), newNCEOp->getLoc(), origOp->getLoc());
        rewriter.eraseOp(newNCEOp);
        return mlir::failure();
    }

    auto newLayerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(newNCEOp);
    if (auto postOpAttr = newLayerWithPostOp.getPostOp()) {
        // NOTE: This is incorrect for all platforms! Current implementation is a hotfix.
        //       isSupportedPostOpAttr should be called instead.
        // TODO: Investigate pipelining and why it causes performance regressions for certain networks
        // when fusing Dequantize into NCEOp. E#234073
        if (!mlir::isa<IE::ReluAttr>(postOpAttr) && !isNonReluPostOpSupported(origOp, quantizedLayerOp)) {
            rewriter.eraseOp(newNCEOp);
            return matchFailed(log, rewriter, origOp,
                               "Op '{0}' has non-ReLU postOp '{1}', preserving Dequantize at loc '{2}'",
                               nceOp->getName(), postOpAttr.getName(), origOp->getLoc());
        }
    }

    log.trace("Replace Op '{0}' '{1}' at loc '{2}' with Op '{3}' '{4}' at loc '{5}'", nceOp->getName(),
              nceOp->getResult(0).getType(), nceOp->getLoc(), newNCEOp->getName(), newNCEOp->getResult(0).getType(),
              newNCEOp->getLoc());
    rewriter.replaceOp(nceOp, newNCEOp->getResult(0));

    for (auto* viewOp : llvm::reverse(chain)) {
        log.trace("Change Op '{0}' at loc '{1}' to '{2}'", viewOp->getName(), viewOp->getLoc(),
                  viewOp->getResult(0).getType());
        inferReturnTypes(viewOp, InferShapedTypeMode::ELEM_TYPE);
    }

    return mlir::success();
}

}  // namespace

mlir::LogicalResult DequantizeWithNCEReLUxRewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.DequantizeOp at loc '{1}'", this->getDebugName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    const auto dequantInputType = origOp.getInput().getType();
    const auto elemType = mlir::cast<mlir::ShapedType>(dequantInputType).getElementType();
    const auto dequantUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    const bool isPerChannel = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    const auto dequantOutputType = origOp.getOutput().getType();
    if (dequantUniformType == nullptr) {
        // NOTE: This is different behaviour to DequantizeWithNCEClampRewriter.
        //       We will fuse Dequantize anyways because accuracy is less important.
        _log.trace("Dequantize is per-axes");
    }

    auto matchResult = findNCEProducer(origOp, rewriter, _log);
    if (mlir::failed(matchResult)) {
        return mlir::failure();
    }
    auto* nceOp = matchResult->nceOp;
    auto& chain = matchResult->chain;

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(nceOp);
    if (quantizedLayerOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Producer Op '{0}' at loc '{1}' is not a quantized layer operation",
                           nceOp->getName(), nceOp->getLoc());
    }

    if (!quantizedLayerOp.isMixPrecisionSupported(!isPerChannel)) {
        return matchFailed(_log, rewriter, origOp, "Producer Op '{0}' at loc '{1}' is not supported", nceOp->getName(),
                           nceOp->getLoc());
    }

    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(nceOp);
    if (layerWithPostOp == nullptr) {
        return matchFailed(_log, rewriter, nceOp,
                           "Producer Op '{0}' at loc '{1}' does not implement LayerWithPostOpInterface",
                           nceOp->getName(), nceOp->getLoc());
    }

    std::optional<Clamp> reluxClamp;
    std::optional<Clamp> existingClamp;
    if (dequantUniformType != nullptr) {
        int64_t levels = 0;
        Clamp dequantClamp = {};
        vpux::getFakeQuantParams(dequantUniformType, levels, dequantClamp.min, dequantClamp.max);
        auto gatheredClamps = gatherClamps(layerWithPostOp, dequantClamp.min, dequantClamp.max);
        existingClamp = gatheredClamps.existingClamp;
        const auto& intersectedClamp = gatheredClamps.intersectedClamp;
        if (intersectedClamp.min > intersectedClamp.max) {
            return matchFailed(_log, rewriter, origOp, "Intersected Clamp is disjointed, min = '{0}', max = '{1}'",
                               intersectedClamp.min, intersectedClamp.max);
        }

        _log.trace("DequantizeOp has clamp range [{0}, {1}]", dequantClamp.min, dequantClamp.max);

        // NOTE: This is ReLU(x)/Clamp(0, x) because hardware supports it, and would be more accurate than ReLU
        if (!isPerChannel && isFloatEqual(intersectedClamp.min, 0.0f)) {
            reluxClamp = intersectedClamp;
            _log.trace("Intersected clamp range [{0}, {1}] is supported", intersectedClamp.min, intersectedClamp.max);
        } else {
            // NOTE: The intersected clamp is unsupported because it's not ReLUx, but we will fuse Dequantize without
            //       clamping for performance reasons
            _log.trace("Intersected clamp range [{0}, {1}] is not supported", intersectedClamp.min,
                       intersectedClamp.max);
        }
    }

    // NOTE: In the case there's no intersected clamp that is ReLUx, check that existing clamp attribute supports
    //       dequantOutputType e.g. MTL/LNL rejects clamp min != 0 for quant-in/quant-out
    if (existingClamp.has_value() && !reluxClamp.has_value()) {
        if (!layerWithPostOp.isSupportedClampProperties(existingClamp->min, existingClamp->max, dequantOutputType,
                                                        logCb)) {
            return matchFailed(_log, rewriter, nceOp,
                               "Op '{0}' at loc '{1}' has clamp attributes [{2}, {3}], but are not "
                               "supported with output type '{4}'",
                               nceOp->getName(), nceOp->getLoc(), existingClamp->min, existingClamp->max,
                               dequantOutputType);
        }
    }

    auto* newNCEOp = cloneAndRetypeNCEOp(nceOp, dequantOutputType, rewriter);
    if (mlir::failed(finalizeFusion(origOp, nceOp, newNCEOp, quantizedLayerOp, chain, rewriter, _log))) {
        return mlir::failure();
    }

    layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(newNCEOp);
    mlir::Value chainTail = computeChainTail(chain, newNCEOp);
    if (reluxClamp.has_value()) {
        if (layerWithPostOp.isSupportedClampProperties(reluxClamp->min, reluxClamp->max, dequantOutputType, logCb)) {
            auto clampAttr = IE::createClampAttr(newNCEOp->getContext(), reluxClamp->min, reluxClamp->max);
            rewriter.modifyOpInPlace(layerWithPostOp, [&]() {
                layerWithPostOp.setClampAttr(clampAttr);
            });
            _log.trace("Fused Clamp at loc '{0}' with range [{1}, {2}]", newNCEOp->getLoc(), reluxClamp->min,
                       reluxClamp->max);
        }
    }

    rewriter.replaceOp(origOp, chainTail);

    return mlir::success();
}

mlir::LogicalResult DequantizeWithNCEClampRewriter::matchAndRewrite(IE::DequantizeOp origOp,
                                                                    mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Got IE.DequantizeOp at loc '{1}'", this->getDebugName(), origOp->getLoc());

    const auto logCb = [&](const formatv_object_base& msg) {
        _log.trace("{0}", msg.str());
    };
    const auto dequantInputType = origOp.getInput().getType();
    const auto elemType = mlir::cast<mlir::ShapedType>(dequantInputType).getElementType();
    const auto dequantUniformType = mlir::dyn_cast<mlir::quant::UniformQuantizedType>(elemType);
    const bool isPerChannel = mlir::isa<mlir::quant::UniformQuantizedPerAxisType>(elemType);
    const auto dequantOutputType = origOp.getOutput().getType();
    if (isPerChannel) {
        return matchFailed(_log, rewriter, origOp,
                           "Per-channel quantized output cannot be expressed by a scalar Clamp");
    }

    if (dequantUniformType == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Unrecognized element type, expected UniformQuantizedType");
    }

    auto matchResult = findNCEProducer(origOp, rewriter, _log);
    if (mlir::failed(matchResult)) {
        return mlir::failure();
    }
    auto* nceOp = matchResult->nceOp;
    auto& chain = matchResult->chain;

    auto quantizedLayerOp = mlir::dyn_cast<IE::QuantizedLayerOpInterface>(nceOp);
    if (quantizedLayerOp == nullptr) {
        return matchFailed(_log, rewriter, origOp, "Producer Op '{0}' at loc '{1}' is not a quantized layer operation",
                           nceOp->getName(), nceOp->getLoc());
    }

    if (!quantizedLayerOp.isMixPrecisionSupported(/*isPReLUSupported*/ true)) {
        return matchFailed(_log, rewriter, origOp, "Producer Op '{0}' at loc '{1}' is not supported", nceOp->getName(),
                           nceOp->getLoc());
    }

    auto layerWithPostOp = mlir::dyn_cast<IE::LayerWithPostOpInterface>(nceOp);
    if (layerWithPostOp == nullptr) {
        return matchFailed(_log, rewriter, nceOp,
                           "Producer Op '{0}' at loc '{1}' does not implement LayerWithPostOpInterface",
                           nceOp->getName(), nceOp->getLoc());
    }

    int64_t levels = 0;
    Clamp dequantClamp = {};
    vpux::getFakeQuantParams(dequantUniformType, levels, dequantClamp.min, dequantClamp.max);
    auto [existingClamp, intersectedClamp] = gatherClamps(layerWithPostOp, dequantClamp.min, dequantClamp.max);
    if (intersectedClamp.min > intersectedClamp.max) {
        return matchFailed(_log, rewriter, origOp, "Intersected Clamp is disjointed, min = '{0}', max = '{1}'",
                           intersectedClamp.min, intersectedClamp.max);
    }

    bool isClampSupportedInNCE = layerWithPostOp.isSupportedClampProperties(intersectedClamp.min, intersectedClamp.max,
                                                                            dequantOutputType, logCb);
    // NOTE: A pre-existing clamp on nceOp must remain representable with dequantOutputType if we can't fuse the
    //       intersected clamp into NCEOp (e.g. MTL/LNL may reject clamp min != 0 for quant-in/quant-out in the future).
    if (!isClampSupportedInNCE && existingClamp.has_value()) {
        if (!layerWithPostOp.isSupportedClampProperties(existingClamp->min, existingClamp->max, dequantOutputType,
                                                        logCb)) {
            return matchFailed(_log, rewriter, nceOp,
                               "nceOp '{0}' at loc '{1}' has clamp attributes [{2}, {3}], but are not "
                               "supported with output type '{4}'",
                               nceOp->getName(), nceOp->getLoc(), existingClamp->min, existingClamp->max,
                               dequantOutputType);
        }
    }

    auto* newNCEOp = cloneAndRetypeNCEOp(nceOp, dequantOutputType, rewriter);
    if (mlir::failed(finalizeFusion(origOp, nceOp, newNCEOp, quantizedLayerOp, chain, rewriter, _log))) {
        return mlir::failure();
    }

    layerWithPostOp = mlir::cast<IE::LayerWithPostOpInterface>(newNCEOp);
    mlir::Value chainTail = computeChainTail(chain, newNCEOp);

    if (isClampSupportedInNCE) {
        auto clampAttr = IE::createClampAttr(newNCEOp->getContext(), intersectedClamp.min, intersectedClamp.max);
        rewriter.modifyOpInPlace(layerWithPostOp, [&]() {
            layerWithPostOp.setClampAttr(clampAttr);
        });
        _log.trace("Fused Clamp at loc '{0}' with range [{1}, {2}]", newNCEOp->getLoc(), intersectedClamp.min,
                   intersectedClamp.max);
    } else {
        _log.trace(
                "Op '{0}' at loc '{1}' does not support fused clamp with range [{2}, {3}] and NCE output type of '{4}'",
                newNCEOp->getName(), newNCEOp->getLoc(), intersectedClamp.min, intersectedClamp.max,
                newNCEOp->getResult(0).getType());
        rewriter.setInsertionPointAfter(newNCEOp);
        auto clampOp = ClampOp::create(rewriter, takeOpLoc(origOp, "as_clamp"), newNCEOp->getResult(0),
                                       rewriter.getF64FloatAttr(intersectedClamp.min),
                                       rewriter.getF64FloatAttr(intersectedClamp.max));
        _log.trace("Created Clamp at loc '{0}' with range [{1}, {2}]", newNCEOp->getLoc(), intersectedClamp.min,
                   intersectedClamp.max);
        // NOTE: The chain references based of nceOp not newNCEOp
        rewriter.replaceAllUsesExcept(newNCEOp->getResult(0), clampOp.getOutput(), clampOp);
        chainTail = chain.empty() ? mlir::Value(clampOp.getOutput()) : chain.front()->getResult(0);
    }

    rewriter.replaceOp(origOp, chainTail);

    return mlir::success();
}
