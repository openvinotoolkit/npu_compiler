//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/concat_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/const_attributes.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_dequantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/const/attributes/content.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/quantization.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/walk_utils.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>
#include <utility>
#include "vpux/compiler/utils/error.hpp"

namespace vpux::IE {
#define GEN_PASS_DECL_MERGEPARALLELFULLYCONNECTED
#define GEN_PASS_DEF_MERGEPARALLELFULLYCONNECTED
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

constexpr int64_t RANK_3D = 3;

struct DynDQConstInputs {
    SmallVector<Const::DeclareOp> inputs;
    SmallVector<Const::DeclareOp> scales;
    SmallVector<Const::DeclareOp> zps;
};

// Convert
//       w1(u2/si8) + s1 + [attrs]                       w2(u2/si8) + s2 + [attrs]
//                 \        |                                     \        |
//                  IE.DynamicDequantize                            IE.DynamicDequantize
//                          |                                              |
//                  IE.AffineReshape(6x2)                         IE.AffineReshape(6x3)
//                          |                                              |
//                  IE.Transpose(2x6)                             IE.Transpose(3x6)
//                          \                                        /
//                           \            Input(1x6)                /
//                            \          /           \             /
//                     IE.FullyConnected(1x2)    IE.FullyConnected(1x3)

// To

//      concat(w1, w2) + concat(s1, s2) + [attrs]
//                 \          |
//                  IE.DynamicDequantize
//                          |
//                  IE.AffineReshape(6x5)
//                          |
//                  IE.Transpose(5x6)       Input(1x6)
//                          \               /
//                           \             /
//                        IE.FullyConnected(1x5)
//                             |          |
//                     IE.Slice(1x2)  IE.Slice(1x3)

class MergeParallelFullyConnected final : public mlir::OpRewritePattern<IE::FullyConnectedOp> {
public:
    MergeParallelFullyConnected(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<IE::FullyConnectedOp>(ctx), _log(std::move(log)) {
        setDebugName("mergeParallelFullyConnected");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::FullyConnectedOp fullyConnectedOp,
                                        mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::Value concatConst(SmallVector<Const::DeclareOp>& constOps, size_t concatDim, mlir::PatternRewriter& rewriter) {
    SmallVector<ShapeRef> inShapes;
    SmallVector<Const::ContentAttr> inputContents;
    inShapes.reserve(constOps.size());
    inputContents.reserve(constOps.size());
    for (auto constOp : constOps) {
        inShapes.push_back(getShape(constOp.getOutput()));
        inputContents.push_back(constOp.getContentAttr());
    }

    const auto offsets = IE::inferConcatOffsets(inShapes, Dim(concatDim), rewriter.getContext());
    const auto contentAttr = Const::createConcatContentAttr(rewriter.getContext(), offsets,
                                                            checked_cast<int64_t>(concatDim), inputContents);

    // Let ConcatAttr::inferOutputType compute the merged type, e.g. it combines per-axis
    // quantization scales/zero-points instead of just reusing the first operand's type.
    const auto outType = contentAttr.getType();
    return Const::DeclareOp::create(rewriter, appendLoc(constOps.front()->getLoc(), "concat_"), outType, contentAttr)
            .getOutput();
}

bool isLastDimConcatable(ShapeRef shape1, ShapeRef shape2) {
    if (shape1.size() != shape2.size()) {
        return false;
    }

    for (auto ind : irange(shape1.size())) {
        if (shape1[Dim(ind)] != shape2[Dim(ind)] && ind != (shape1.size() - 1)) {
            return false;
        }
    }

    return true;
};

template <class ConcreteOp>
bool isOutputCompatible(SmallVector<ConcreteOp>& ops) {
    auto refOp = ops.back();
    auto refElemType = mlir::cast<vpux::NDTypeInterface>(refOp->getResult(0).getType()).getElementType();
    auto refShape = getShape(refOp->getResult(0));
    for (auto ind : irange(ops.size() - 1)) {
        auto op = ops[ind];
        if (!isLastDimConcatable(refShape, getShape(op->getResult(0)))) {
            return false;
        }

        auto outElementType = mlir::cast<vpux::NDTypeInterface>(op->getResult(0).getType()).getElementType();
        if (outElementType != refElemType) {
            return false;
        }
    }
    return true;
}

std::optional<SmallVector<IE::FullyConnectedOp>> getFullyConnectedOpWithSameAttr(mlir::Value parent) {
    SmallVector<IE::FullyConnectedOp> fullyConnectedOps;
    for (auto user : parent.getUsers()) {
        auto fullConnected = mlir::dyn_cast<IE::FullyConnectedOp>(user);
        if (fullConnected == nullptr) {
            return std::nullopt;
        }
        if (fullConnected.getInput() != parent) {
            return std::nullopt;
        }
        fullyConnectedOps.push_back(fullConnected);
    }

    // need at least two fullyConnected
    if (fullyConnectedOps.size() < 2) {
        return std::nullopt;
    }

    if (!isOutputCompatible<IE::FullyConnectedOp>(fullyConnectedOps)) {
        return std::nullopt;
    }

    auto hasAllButLastDimOne = [](ShapeRef shape) {
        const auto isOne = [](auto dim) {
            return dim == 1;
        };
        if (shape.empty()) {
            return false;
        }

        return std::all_of(shape.begin(), std::prev(shape.end()), isOne);
    };

    for (auto fullConnected : fullyConnectedOps) {
        auto shape = getShape(fullConnected.getOutput());
        if (!hasAllButLastDimOne(shape)) {
            return std::nullopt;
        }
        // TODO: could support if the bias data is const.
        if (fullConnected.getBias() != nullptr) {
            return std::nullopt;
        }
    }

    // Sorting those ops is required for deterministic compilation as the order of getUsers
    // might be different between compilations depending on preceding passes
    llvm::sort(fullyConnectedOps, [](IE::FullyConnectedOp fc1, IE::FullyConnectedOp fc2) {
        return !fc1->isBeforeInBlock(fc2);
    });

    return fullyConnectedOps;
}

namespace AffineReshapeTransposeOrder {
std::optional<SmallVector<IE::TransposeOp>> getTransposeOpWithSameAttr(
        ArrayRef<IE::FullyConnectedOp> fullyConnectedOps) {
    SmallVector<IE::TransposeOp> transposeOps;
    for (auto fullyConnected : fullyConnectedOps) {
        auto transpose = mlir::dyn_cast<IE::TransposeOp>(fullyConnected.getWeights().getDefiningOp());
        if (transpose == nullptr || !transpose->hasOneUse() || !transpose.getOrderValue().has_value()) {
            return std::nullopt;
        }
        transposeOps.push_back(transpose);
    }

    // should have same order value
    auto refTranspose = transposeOps.back();
    auto refOrderValue = refTranspose.getOrderValue().value();
    for (auto transpose : transposeOps) {
        if (transpose.getOrderValue().value() != refOrderValue) {
            return std::nullopt;
        }
    }

    return transposeOps;
}

std::optional<SmallVector<IE::AffineReshapeOp>> getAffineReshapeOpWithSameAttr(ArrayRef<IE::TransposeOp> transposeOps) {
    SmallVector<IE::AffineReshapeOp> affineReshapeOps;
    for (auto transpose : transposeOps) {
        auto affineReshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(transpose.getInput().getDefiningOp());
        if (affineReshape == nullptr || !affineReshape->hasOneUse()) {
            return std::nullopt;
        }
        affineReshapeOps.push_back(affineReshape);
    }

    // should have the same dim mapping
    auto refAffineReshape = affineReshapeOps.back();
    auto refDimMapping = refAffineReshape.getDimMapping();
    for (auto affineReshape : affineReshapeOps) {
        if (getShape(affineReshape.getOutput()).size() != RANK_3D - 1 ||
            affineReshape.getDimMapping() != refDimMapping) {
            return std::nullopt;
        }
    }

    return affineReshapeOps;
}
}  // namespace AffineReshapeTransposeOrder

namespace TransposeAffineReshapeOrder {
std::optional<SmallVector<IE::AffineReshapeOp>> getAffineReshapeOpWithSameAttr(ArrayRef<IE::FullyConnectedOp> fcOps) {
    SmallVector<IE::AffineReshapeOp> affineReshapeOps;
    for (auto fc : fcOps) {
        auto affineReshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(fc.getWeights().getDefiningOp());
        if (affineReshape == nullptr || !affineReshape->hasOneUse()) {
            return std::nullopt;
        }
        affineReshapeOps.push_back(affineReshape);
    }

    // should have the same dim mapping
    auto refAffineReshape = affineReshapeOps.back();
    auto refDimMapping = refAffineReshape.getDimMapping();
    for (auto affineReshape : affineReshapeOps) {
        if (getShape(affineReshape.getOutput()).size() != RANK_3D - 1 ||
            affineReshape.getDimMapping() != refDimMapping) {
            return std::nullopt;
        }
    }

    return affineReshapeOps;
}

std::optional<SmallVector<IE::TransposeOp>> getTransposeOpWithSameAttr(ArrayRef<IE::AffineReshapeOp> affineReshapeOps) {
    SmallVector<IE::TransposeOp> transposeOps;
    for (auto reshape : affineReshapeOps) {
        auto transpose = mlir::dyn_cast_if_present<IE::TransposeOp>(reshape.getInput().getDefiningOp());
        if (transpose == nullptr || !transpose->hasOneUse() || !transpose.getOrderValue().has_value()) {
            return std::nullopt;
        }
        transposeOps.push_back(transpose);
    }

    // should have same order value
    auto refTranspose = transposeOps.back();
    auto refOrderValue = refTranspose.getOrderValue().value();
    for (auto transpose : transposeOps) {
        if (transpose.getOrderValue().value() != refOrderValue) {
            return std::nullopt;
        }
    }

    return transposeOps;
}
}  // namespace TransposeAffineReshapeOrder

template <class ConcreteOp>
std::optional<SmallVector<IE::DynamicDequantizeOp>> getDynamicDequantizeOpWithSameAttr(ArrayRef<ConcreteOp> ops) {
    SmallVector<IE::DynamicDequantizeOp> dynamicDequantOps;
    for (auto op : ops) {
        if (!op->hasOneUse()) {
            return std::nullopt;
        }

        auto dynamicDequant = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(op.getInput().getDefiningOp());
        if (dynamicDequant == nullptr || !dynamicDequant->hasOneUse() || !dynamicDequant.getDstElemType()) {
            return std::nullopt;
        }

        dynamicDequantOps.push_back(dynamicDequant);
    }

    auto refDynamicDequant = dynamicDequantOps.back();
    auto refDstElemType = refDynamicDequant.getDstElemType();
    auto refInElemType = mlir::cast<vpux::NDTypeInterface>(refDynamicDequant.getInput().getType()).getElementType();
    auto refSyntheticAttr = refDynamicDequant->getAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR);
    auto refWeightsImportAttr = refDynamicDequant->getAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR);
    const bool refHasZp = refDynamicDequant.getZp() != nullptr;
    for (auto dynamicDequant : dynamicDequantOps) {
        // Input element type must match too: concatConst() reuses the first operand's type for
        // the merged constant, so mixing e.g. si8 and ui4 inputs would silently misinterpret data.
        const auto inElemType = mlir::cast<vpux::NDTypeInterface>(dynamicDequant.getInput().getType()).getElementType();
        if (dynamicDequant.getDstElemType() != refDstElemType || inElemType != refInElemType ||
            dynamicDequant->getAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR) != refWeightsImportAttr ||
            dynamicDequant->getAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR) != refSyntheticAttr ||
            ((dynamicDequant.getZp() != nullptr) != refHasZp)) {
            return std::nullopt;
        }
    }

    return dynamicDequantOps;
}

DynDQConstInputs getDynamicDequantizeConstInputs(SmallVector<IE::DynamicDequantizeOp>& dynamicDequantizeOps) {
    DynDQConstInputs dynDQConstInputs;
    bool hasZp = false;
    bool hasNoZp = false;
    for (auto dynamicDequantize : dynamicDequantizeOps) {
        auto input = mlir::dyn_cast_if_present<Const::DeclareOp>(dynamicDequantize.getInput().getDefiningOp());
        auto scale = mlir::dyn_cast_if_present<Const::DeclareOp>(dynamicDequantize.getScale().getDefiningOp());
        const bool opHasZp = dynamicDequantize.getZp() != nullptr;
        Const::DeclareOp zp = nullptr;
        if (opHasZp) {
            hasZp = true;
            zp = mlir::dyn_cast_if_present<Const::DeclareOp>(dynamicDequantize.getZp().getDefiningOp());
        } else {
            hasNoZp = true;
        }
        // If one of the DynamicDequantizeOp operands is not a Const::DeclareOp we cannot merge the parallel
        // FullyConnected ops. We clear dynDQConstInputs variable and later check for empty SmallVector in order to
        // safely return matchFailed.
        if (input == nullptr || scale == nullptr || (opHasZp && zp == nullptr)) {
            dynDQConstInputs.inputs.clear();
            dynDQConstInputs.scales.clear();
            dynDQConstInputs.zps.clear();
            return dynDQConstInputs;
        }
        dynDQConstInputs.inputs.push_back(input);
        dynDQConstInputs.scales.push_back(scale);
        if (zp != nullptr) {
            dynDQConstInputs.zps.push_back(zp);
        }
    }

    if (hasZp && hasNoZp) {
        dynDQConstInputs.inputs.clear();
        dynDQConstInputs.scales.clear();
        dynDQConstInputs.zps.clear();
        return dynDQConstInputs;
    }

    return dynDQConstInputs;
}

IE::DynamicDequantizeOp createDynamicDequantize(SmallVector<IE::DynamicDequantizeOp>& dynamicDequantizeOps,
                                                size_t inputConcatDim, mlir::PatternRewriter& rewriter) {
    auto dynDQConstInputs = getDynamicDequantizeConstInputs(dynamicDequantizeOps);

    if (dynDQConstInputs.inputs.empty()) {
        return nullptr;
    }

    // Merged input shape, derived from the individual input shapes so that scale/zp validation
    // below doesn't need the (potentially large) concatenated input constant materialized yet.
    auto mergedInShapeVec = SmallVector<int64_t>(getShape(dynDQConstInputs.inputs.front().getOutput()).raw().begin(),
                                                 getShape(dynDQConstInputs.inputs.front().getOutput()).raw().end());
    mergedInShapeVec[inputConcatDim] = 0;
    for (auto inConst : dynDQConstInputs.inputs) {
        mergedInShapeVec[inputConcatDim] += getShape(inConst.getOutput())[Dim(inputConcatDim)];
    }
    const auto mergedInShape = Shape(mergedInShapeVec);

    mlir::Value scaleValue = dynDQConstInputs.scales.front().getOutput();
    size_t scaleConcatDim = 0;
    bool scaleNeedsConcat = false;
    if (getShape(scaleValue).totalSize() == 1) {
        SmallVector<float> scaleData;
        scaleData.reserve(dynDQConstInputs.scales.size());
        for (auto scaleConst : dynDQConstInputs.scales) {
            if (!IE::isBaseContentSplat(scaleConst)) {
                return nullptr;
            }
            scaleData.push_back(IE::getConst(scaleConst).front());
        }
        if (!llvm::all_of(llvm::ArrayRef<float>(scaleData).drop_front(), [&](float v) {
                return v == scaleData.front();
            })) {
            return nullptr;
        }
    } else {
        const auto refScaleShape = getShape(scaleValue);
        const auto nonOneDims = vpux::getNonOneDim(refScaleShape);

        if (nonOneDims.size() == 1) {
            scaleConcatDim = checked_cast<size_t>(nonOneDims.front().ind());
        } else {
            // For multi-dimensional (e.g., per-group) scales, concatenate along the
            // same dimension used for weight concatenation.
            if (inputConcatDim >= refScaleShape.size()) {
                return nullptr;
            }
            scaleConcatDim = inputConcatDim;
        }

        for (auto scaleConst : dynDQConstInputs.scales) {
            const auto currentScaleShape = getShape(scaleConst.getOutput());
            if (currentScaleShape.totalSize() == 1 || currentScaleShape.size() != refScaleShape.size()) {
                return nullptr;
            }

            for (auto dimIdx : irange(refScaleShape.size())) {
                if (dimIdx == scaleConcatDim) {
                    continue;
                }
                if (currentScaleShape[Dim(dimIdx)] != refScaleShape[Dim(dimIdx)]) {
                    return nullptr;
                }
            }
        }

        scaleNeedsConcat = true;
    }

    mlir::Value zpValue = nullptr;
    if (!dynDQConstInputs.zps.empty()) {
        SmallVector<float> zpData;
        for (auto zpConst : dynDQConstInputs.zps) {
            if (!IE::isBaseContentSplat(zpConst)) {
                return nullptr;
            }
            zpData.push_back(IE::getConst(zpConst).front());
        }
        if (!llvm::all_of(llvm::ArrayRef<float>(zpData).drop_front(), [&](const float zp) {
                return zp == zpData.front();
            })) {
            return nullptr;
        }

        // Keep zero-point shape compatible with the merged input shape.
        // DynamicDequantize verifier requires same rank and broadcastable dims.
        const auto zpShape = getShape(dynDQConstInputs.zps.front().getOutput());
        if (mergedInShape.size() != zpShape.size()) {
            return nullptr;
        }
        for (auto d : irange(mergedInShape.size())) {
            if (zpShape[Dim(d)] > 1 && zpShape[Dim(d)] != mergedInShape[Dim(d)]) {
                return nullptr;
            }
        }
        zpValue = dynDQConstInputs.zps.front().getOutput();
    }

    // All validation above passed; only now materialize the (potentially large) concatenated constants.
    auto concatInConst = concatConst(dynDQConstInputs.inputs, inputConcatDim, rewriter);
    if (scaleNeedsConcat) {
        scaleValue = concatConst(dynDQConstInputs.scales, scaleConcatDim, rewriter);
    }

    auto refOp = dynamicDequantizeOps.front();
    auto newDynDQ = IE::DynamicDequantizeOp::create(rewriter, appendLoc(refOp->getLoc(), "concat_"), concatInConst,
                                                    scaleValue, zpValue, refOp.getDstElemType());

    if (dynamicDequantizeOps.front()->hasAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR)) {
        newDynDQ->setAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR,
                          dynamicDequantizeOps.front()->getAttr(IE::SYNTHETIC_DYN_DEQUANT_ATTR));
    }

    if (dynamicDequantizeOps.front()->hasAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR)) {
        newDynDQ->setAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR,
                          dynamicDequantizeOps.front()->getAttr(IE::WEIGHTS_IMPORT_DYN_DEQUANT_ATTR));
    }

    return newDynDQ;
}

// Creates a merged IE.FullyConnectedOp from N parallel FC ops whose weights are direct
// Const::DeclareOp values. Weights are concatenated along axis 0 (output-channel axis)
// in the order given by fcOrder, which must match the input ordering of the downstream
// consumer (e.g. a Concat) to preserve output semantics. The caller is responsible for
// replacing or slicing the merged FC output to recover individual FC results.
//
// fcOrder must be sorted to match the concatenation semantics of the downstream consumer.
IE::FullyConnectedOp mergeFCForDirectWeightsOrder(ArrayRef<IE::FullyConnectedOp> fcOrder, IE::FullyConnectedOp origOp,
                                                  mlir::PatternRewriter& rewriter) {
    SmallVector<mlir::Value> weights;
    for (auto fc : fcOrder) {
        weights.push_back(fc.getWeights());
    }
    auto mergedWeights = IE::ConcatOp::create(rewriter, appendLoc(origOp->getLoc(), "concat_weights"), weights, Dim(0));
    return IE::FullyConnectedOp::create(rewriter, appendLoc(origOp->getLoc(), "concat_"), origOp.getInput(),
                                        mergedWeights.getOutput(), nullptr);
}

mlir::FailureOr<IE::FullyConnectedOp> mergeFCForDynDQReshapeTransposeOrder(
        ArrayRef<IE::FullyConnectedOp> fullyConnectedOps, IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter,
        Logger log) {
    auto validTransposeOps = AffineReshapeTransposeOrder::getTransposeOpWithSameAttr(fullyConnectedOps);
    if (!validTransposeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid transpose operations");
    }
    auto transposeOps = validTransposeOps.value();

    auto validAffineReshapeOps = AffineReshapeTransposeOrder::getAffineReshapeOpWithSameAttr(transposeOps);
    if (!validAffineReshapeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid affineReshape operations");
    }
    auto affineReshapeOps = validAffineReshapeOps.value();

    auto validDynamicDequantizeOps = getDynamicDequantizeOpWithSameAttr<IE::AffineReshapeOp>(affineReshapeOps);
    if (!validDynamicDequantizeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid dynamic dequantize operations");
    }
    auto dynamicDequantizeOps = validDynamicDequantizeOps.value();
    const auto inputConcatDim = getShape(dynamicDequantizeOps.front().getInput()).size() - 1;
    auto newDynamicDequantize = createDynamicDequantize(dynamicDequantizeOps, inputConcatDim, rewriter);
    if (newDynamicDequantize == nullptr) {
        return matchFailed(log, rewriter, origOp,
                           "Unable to create merged DynamicDequantize op for AffineReshape-Transpose path");
    }

    const auto newDynamicDequantizeOutShape = getShape(newDynamicDequantize.getOutput());
    SmallVector<int64_t> reshapeOut{getShape(affineReshapeOps.front().getOutput()).raw().front(),
                                    newDynamicDequantizeOutShape.raw().back()};
    const auto reshapeOutAttr = getIntArrayAttr(origOp->getContext(), reshapeOut);
    auto newAffineReshape = IE::AffineReshapeOp::create(
            rewriter, appendLoc(affineReshapeOps.front()->getLoc(), "concat_"), newDynamicDequantize.getOutput(),
            affineReshapeOps.front().getDimMapping(), reshapeOutAttr);

    auto newTranspose =
            IE::TransposeOp::create(rewriter, appendLoc(transposeOps.front()->getLoc(), "concat_"),
                                    newAffineReshape.getOutput(), nullptr, transposeOps.front().getOrderValueAttr());
    auto newFullyConnected =
            IE::FullyConnectedOp::create(rewriter, appendLoc(origOp->getLoc(), "concat_"), origOp.getInput(),
                                         newTranspose.getOutput(), origOp.getBias());

    return newFullyConnected;
}

mlir::FailureOr<IE::FullyConnectedOp> mergeFCForDynDQTransposeReshapeOrder(
        ArrayRef<IE::FullyConnectedOp> fullyConnectedOps, IE::FullyConnectedOp origOp, mlir::PatternRewriter& rewriter,
        Logger log) {
    auto validAffineReshapeOps = TransposeAffineReshapeOrder::getAffineReshapeOpWithSameAttr(fullyConnectedOps);
    if (!validAffineReshapeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid affineReshape operations");
    }
    auto affineReshapeOps = validAffineReshapeOps.value();

    auto validTransposeOps = TransposeAffineReshapeOrder::getTransposeOpWithSameAttr(affineReshapeOps);
    if (!validTransposeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid transpose operations");
    }
    auto transposeOps = validTransposeOps.value();
    auto validDynamicDequantizeOps = getDynamicDequantizeOpWithSameAttr<IE::TransposeOp>(transposeOps);
    if (!validDynamicDequantizeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid dynamic dequantize operations");
    }
    auto dynamicDequantizeOps = validDynamicDequantizeOps.value();
    const auto inputConcatDim = getShape(dynamicDequantizeOps.front().getInput()).size() - 1;
    auto newDynamicDequantize = createDynamicDequantize(dynamicDequantizeOps, inputConcatDim, rewriter);
    if (newDynamicDequantize == nullptr) {
        return matchFailed(log, rewriter, origOp,
                           "Unable to create merged DynamicDequantize op for Transpose-AffineReshape path");
    }

    auto newTranspose = IE::TransposeOp::create(rewriter, appendLoc(transposeOps.front()->getLoc(), "concat_"),
                                                newDynamicDequantize.getOutput(), nullptr,
                                                transposeOps.front().getOrderValueAttr());

    const auto newTransposeOutShape = getShape(newTranspose.getOutput());
    SmallVector<int64_t> reshapeOut{newTransposeOutShape.raw().front(),
                                    getShape(affineReshapeOps.front().getOutput()).raw().back()};
    const auto reshapeOutAttr = getIntArrayAttr(origOp->getContext(), reshapeOut);
    auto newAffineReshape = IE::AffineReshapeOp::create(
            rewriter, appendLoc(affineReshapeOps.front()->getLoc(), "concat_"), newTranspose.getOutput(),
            affineReshapeOps.front().getDimMapping(), reshapeOutAttr);

    auto newFullyConnected =
            IE::FullyConnectedOp::create(rewriter, appendLoc(origOp->getLoc(), "concat_"), origOp.getInput(),
                                         newAffineReshape.getOutput(), origOp.getBias());

    return newFullyConnected;
}

mlir::FailureOr<IE::FullyConnectedOp> mergeFCForDynDQReshapeOrder(ArrayRef<IE::FullyConnectedOp> fullyConnectedOps,
                                                                  IE::FullyConnectedOp origOp,
                                                                  mlir::PatternRewriter& rewriter, Logger log) {
    auto validAffineReshapeOps = TransposeAffineReshapeOrder::getAffineReshapeOpWithSameAttr(fullyConnectedOps);
    if (!validAffineReshapeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid affineReshape operations");
    }
    auto affineReshapeOps = validAffineReshapeOps.value();

    auto validDynamicDequantizeOps = getDynamicDequantizeOpWithSameAttr<IE::AffineReshapeOp>(affineReshapeOps);
    if (!validDynamicDequantizeOps.has_value()) {
        return matchFailed(log, rewriter, origOp, "Invalid dynamic quantize operations");
    }
    auto dynamicDequantizeOps = validDynamicDequantizeOps.value();
    auto newDynamicDequantize = createDynamicDequantize(dynamicDequantizeOps, /*inputConcatDim=*/0, rewriter);
    if (newDynamicDequantize == nullptr) {
        return matchFailed(log, rewriter, origOp,
                           "Unable to create new DynamicDequantize op, DynamicDequantize op has non const inputs.");
    }
    const auto newDynamicDequantizeOutShape = getShape(newDynamicDequantize.getOutput());
    SmallVector<int64_t> reshapeOut{newDynamicDequantizeOutShape.raw().front(),
                                    getShape(affineReshapeOps.front().getOutput()).raw().back()};
    const auto reshapeOutAttr = getIntArrayAttr(origOp->getContext(), reshapeOut);
    auto newAffineReshape = IE::AffineReshapeOp::create(
            rewriter, appendLoc(affineReshapeOps.front()->getLoc(), "concat_"), newDynamicDequantize.getOutput(),
            affineReshapeOps.front().getDimMapping(), reshapeOutAttr);

    auto newFullyConnected =
            IE::FullyConnectedOp::create(rewriter, appendLoc(origOp->getLoc(), "concat_"), origOp.getInput(),
                                         newAffineReshape.getOutput(), origOp.getBias());

    return newFullyConnected;
}

mlir::LogicalResult MergeParallelFullyConnected::matchAndRewrite(IE::FullyConnectedOp fullyConnectedOp,
                                                                 mlir::PatternRewriter& rewriter) const {
    _log.debug("[{0}] Got FullyConnected layer at '{1}'", fullyConnectedOp->getName(), fullyConnectedOp->getLoc());
    auto nestedLog = _log.nest();

    if (fullyConnectedOp->getUses().empty()) {
        return mlir::failure();  // operation is already handled
    }

    auto validFullyConnectedOps = getFullyConnectedOpWithSameAttr(fullyConnectedOp.getInput());
    if (!validFullyConnectedOps.has_value()) {
        return matchFailed(_log, rewriter, fullyConnectedOp, "Invalid fullyConnected operations");
    }
    auto fullyConnectedOps = validFullyConnectedOps.value();

    auto parentIsAffineReshapeWithDynamicDequantizeOp = [](IE::FullyConnectedOp fcOp) {
        auto lhsOp = fcOp.getWeights().getDefiningOp();
        if (auto reshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(lhsOp)) {
            auto dynamicDequant =
                    mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(reshape.getInput().getDefiningOp());
            return dynamicDequant != nullptr;
        }
        return false;
    };
    auto parentIsTransposeWithDynDQAffineReshapeInput = [](IE::FullyConnectedOp fcOp) {
        auto lhsOp = fcOp.getWeights().getDefiningOp();
        auto transpose = mlir::dyn_cast_if_present<IE::TransposeOp>(lhsOp);
        if (transpose == nullptr) {
            return false;
        }

        auto reshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(transpose.getInput().getDefiningOp());
        if (reshape == nullptr) {
            return false;
        }

        auto dynamicDequant = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(reshape.getInput().getDefiningOp());
        return dynamicDequant != nullptr;
    };
    auto parentIsAffineReshapeWithDynDQTransposeInput = [](IE::FullyConnectedOp fcOp) {
        auto lhsOp = fcOp.getWeights().getDefiningOp();
        auto reshape = mlir::dyn_cast_if_present<IE::AffineReshapeOp>(lhsOp);
        if (reshape == nullptr) {
            return false;
        }

        auto transpose = mlir::dyn_cast_if_present<IE::TransposeOp>(reshape.getInput().getDefiningOp());
        if (transpose == nullptr) {
            return false;
        }

        auto dynamicDequant = mlir::dyn_cast_if_present<IE::DynamicDequantizeOp>(transpose.getInput().getDefiningOp());
        return dynamicDequant != nullptr;
    };
    // Helper: trace FC output → optional single AffineReshape → Concat.
    auto getConcatThroughReshape = [](IE::FullyConnectedOp fc) -> IE::ConcatOp {
        mlir::Value val = fc.getOutput();
        if (!val.hasOneUse()) {
            return nullptr;
        }
        auto* user = *val.getUsers().begin();
        if (auto concat = mlir::dyn_cast<IE::ConcatOp>(user)) {
            return concat;
        }
        if (auto reshape = mlir::dyn_cast<IE::AffineReshapeOp>(user)) {
            if (!reshape->hasOneUse()) {
                return nullptr;
            }
            return mlir::dyn_cast<IE::ConcatOp>(*reshape->getUsers().begin());
        }
        return nullptr;
    };

    // Detect the direct-const-weights + Concat-elimination path.
    // Only applies when ALL of the following hold:
    //   1. Every FC weight is a direct Const::DeclareOp (no preprocessing).
    //   2. Every FC output (through an optional AffineReshape) feeds the same Concat.
    //   3. That Concat has a valid single contiguous axis.
    // When this path fires the Concat is replaced by a Reshape; no Slice ops are emitted.
    auto parentIsConstOp = [](IE::FullyConnectedOp fcOp) {
        return mlir::dyn_cast_if_present<Const::DeclareOp>(fcOp.getWeights().getDefiningOp()) != nullptr;
    };
    bool isDirectConst = llvm::all_of(fullyConnectedOps, parentIsConstOp);

    IE::ConcatOp sharedConcat = isDirectConst ? getConcatThroughReshape(fullyConnectedOps.front()) : nullptr;
    bool allFeedSameConcat = sharedConcat != nullptr && IE::getConcatAxis(sharedConcat).has_value();
    if (allFeedSameConcat) {
        for (auto fc : fullyConnectedOps) {
            if (getConcatThroughReshape(fc) != sharedConcat) {
                allFeedSameConcat = false;
                break;
            }
        }
        // Verify the Concat has no extra inputs beyond the FC group to avoid silently
        // dropping unrelated Concat inputs when the op is replaced.
        if (allFeedSameConcat && sharedConcat.getInputs().size() != fullyConnectedOps.size()) {
            allFeedSameConcat = false;
        }
    }
    const bool directConstWithConcat = isDirectConst && allFeedSameConcat;

    auto maybeDynDQReshapeTranspose = llvm::all_of(fullyConnectedOps, parentIsTransposeWithDynDQAffineReshapeInput);
    auto maybeDynDQTransposeReshape = llvm::all_of(fullyConnectedOps, parentIsAffineReshapeWithDynDQTransposeInput);
    auto maybeDynDQReshape = llvm::all_of(fullyConnectedOps, parentIsAffineReshapeWithDynamicDequantizeOp);

    if (!maybeDynDQReshapeTranspose && !maybeDynDQTransposeReshape && !maybeDynDQReshape && !directConstWithConcat) {
        nestedLog.debug(
                "At least one parent is neither DynamicDequantize-based weights nor direct Const feeding a Concat");
        return mlir::failure();
    }

    // Fast path: direct const weights + Concat elimination (no Slices needed).
    //
    // Pattern (direct const weights, all FC outputs → optional AffineReshape → same Concat):
    //
    //  cst_w1(N1xK)  cst_w2(N2xK)
    //       |               |
    //       \    Input(BxK) /
    //        \   /       \ /
    //    FC(BxN1)      FC(BxN2)
    //        |               |
    //  AffineReshape   AffineReshape        ← optional
    //   (Bx1xN1)        (Bx1xN2)
    //         \              /
    //       Concat(axis=0) → (2xBxN_total/2)
    //
    // Becomes:
    //
    //  Input(BxK)    Concat(cst_w1, cst_w2, axis=0)
    //           \           ((N1+N2)xK)
    //            \          /
    //          FC(Bx(N1+N2))
    //                |
    //      Reshape → (2xBxN_total/2)   ← Concat eliminated
    //
    if (directConstWithConcat) {
        // Re-order FCs to match the input ordering of sharedConcat so that the merged
        // weight tensor preserves the same output ordering as the original Concat.
        SmallVector<IE::FullyConnectedOp> orderedFCOps;
        orderedFCOps.reserve(sharedConcat.getInputs().size());
        for (auto concatInput : sharedConcat.getInputs()) {
            mlir::Value v = concatInput;
            if (auto reshape = concatInput.getDefiningOp<IE::AffineReshapeOp>()) {
                v = reshape.getInput();
            }
            auto fc = v.getDefiningOp<IE::FullyConnectedOp>();
            VPUX_THROW_WHEN(fc == nullptr, "Expected FullyConnectedOp feeding Concat input");
            orderedFCOps.push_back(fc);
        }

        auto concatOutShape = mlir::cast<mlir::RankedTensorType>(sharedConcat.getOutput().getType()).getShape();
        auto shapeAttr = getIntArrayAttr(rewriter.getContext(), concatOutShape);
        nestedLog.trace("Direct-const FC + Concat eliminated at '{0}'", sharedConcat->getLoc());
        auto mergedFC = mergeFCForDirectWeightsOrder(orderedFCOps, fullyConnectedOp, rewriter);
        auto reshaped = IE::ReshapeOp::create(rewriter, appendLoc(mergedFC->getLoc(), "reshape_merged"),
                                              mergedFC.getOutput(), shapeAttr)
                                .getResult();
        rewriter.replaceOp(sharedConcat, reshaped);
        _log.debug("Merge parallel fully connected (concat elimination) successful");
        return mlir::success();
    }

    mlir::FailureOr<IE::FullyConnectedOp> mergedFC;
    if (maybeDynDQReshape) {
        // DynDQ - AffineReshape - FullyConnected
        mergedFC = mergeFCForDynDQReshapeOrder(fullyConnectedOps, fullyConnectedOp, rewriter, _log);
    } else if (maybeDynDQReshapeTranspose) {
        // DynDQ - AffineReshape - Transpose - FullyConnected
        mergedFC = mergeFCForDynDQReshapeTransposeOrder(fullyConnectedOps, fullyConnectedOp, rewriter, _log);
    } else {
        // DynDQ - Transpose - AffineReshape - FullyConnected
        mergedFC = mergeFCForDynDQTransposeReshapeOrder(fullyConnectedOps, fullyConnectedOp, rewriter, _log);
    }

    if (mlir::failed(mergedFC)) {
        nestedLog.debug("Failed to merge parallel FullyConnected ops");
        return mlir::failure();
    }

    nestedLog.trace("New merged FullyConnected op = {0}", mergedFC.value());

    int64_t offset = 0;
    SmallVector<IE::SliceOp> slices;

    // Create Slice ops for each original FullyConnected op
    for (auto p : fullyConnectedOps | indexed) {
        auto fullyConnected = p.value();
        auto shape = getShape(fullyConnected.getOutput());
        Shape offsets(shape.size());
        offsets[Dim(shape.size() - 1)] = offset;
        offset += shape[Dim(shape.size() - 1)];

        nestedLog.trace("Slice output of FullyConnected ops, idx = {0}, offsets = {1}", p.index(), offsets);
        auto slice = IE::SliceOp::create(rewriter, appendLoc(fullyConnected->getLoc(), "slice_{0}", p.index()),
                                         mergedFC.value().getOutput(), getIntArrayAttr(rewriter.getContext(), offsets),
                                         getIntArrayAttr(rewriter.getContext(), shape.raw()));

        slices.push_back(slice);
    }

    // Replace the original FullyConnected ops
    for (auto p : fullyConnectedOps | indexed) {
        auto fullyConnected = p.value();
        auto slice = slices[p.index()];

        nestedLog.trace("Replace FC = {0}, with Slice output {1}", fullyConnected, slice);
        rewriter.replaceAllOpUsesWith(fullyConnected, slice->getResult(0));
    }

    _log.debug("Merge parallel fully connected operation successful");
    return mlir::success();
}

//
// MergeParallelFullyConnectedPass
//

class MergeParallelFullyConnectedPass final :
        public IE::impl::MergeParallelFullyConnectedBase<MergeParallelFullyConnectedPass> {
public:
    explicit MergeParallelFullyConnectedPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnFunc
//

void MergeParallelFullyConnectedPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<MergeParallelFullyConnected>(&ctx, _log);

    collectOpsAndApplyPatterns(func, std::move(patterns));

    mlir::RewritePatternSet concatPatterns(&ctx);
    IE::ConcatOp::getCanonicalizationPatterns(concatPatterns, &ctx);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(concatPatterns), getDefaultGreedyRewriteConfig()))) {
        signalPassFailure();
    }
}

}  // namespace

//
//  createMergeParallelFullyConnectedPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createMergeParallelFullyConnectedPass(Logger log) {
    return std::make_unique<MergeParallelFullyConnectedPass>(log);
}
