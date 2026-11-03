//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/arithmetic.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/data_type.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/normalization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/utils/broadcast_utils.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/attributes_utils.hpp"
#include "vpux/compiler/utils/infer_output_shape.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <cmath>

namespace vpux::IE {
#define GEN_PASS_DECL_FUSENORMALIZEL2TORMS
#define GEN_PASS_DEF_FUSENORMALIZEL2TORMS
#include "vpux/compiler/dialect/IE/passes.hpp.inc"
}  // namespace vpux::IE

using namespace vpux;

namespace {

//
// FuseNormalizeL2ToRMSPass
//

class FuseNormalizeL2ToRMSPass final : public IE::impl::FuseNormalizeL2ToRMSBase<FuseNormalizeL2ToRMSPass> {
public:
    explicit FuseNormalizeL2ToRMSPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

mlir::FailureOr<int64_t> getReducedSize(ArrayRef<int64_t> reduceAxes, vpux::ShapeRef reduceInputShape) {
    int64_t reduceSize = 1;
    const auto rank = checked_cast<int64_t>(reduceInputShape.size());
    for (const auto& axis : reduceAxes) {
        const auto normalizedAxis = axis < 0 ? axis + rank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= rank) {
            return mlir::failure();
        }
        reduceSize *= reduceInputShape[Dim(normalizedAxis)];
    }
    return reduceSize;
}

mlir::FailureOr<mlir::Value> broadcastAndRescaleGamma(mlir::OpBuilder& builder, Const::DeclareOp constOp,
                                                      int64_t targetSize, double rescaleFactor,
                                                      mlir::Type targetElemType, mlir::Location loc) {
    auto shape = Shape(getShape(constOp->getResult(0)).raw());
    auto nonOneDims = getNonOneDim(shape);
    if (nonOneDims.size() > 1) {
        return mlir::failure();
    }

    auto dimSize = nonOneDims.empty() ? int64_t(1) : shape[nonOneDims.back()];
    if (!isBroadcastable(dimSize, targetSize)) {
        return mlir::failure();
    }

    auto contentAttrBuilder = constOp.transformContentAttr();
    if (shape.empty()) {
        // Treat scalar as [1] so broadcast uses a valid dimension.
        contentAttrBuilder = contentAttrBuilder.reshape(ShapeRef({1}));
        shape = Shape{1};
    }

    auto broadcastDim = nonOneDims.empty() ? Dim(shape.size() - 1) : nonOneDims.back();
    if (dimSize != targetSize) {
        contentAttrBuilder = contentAttrBuilder.broadcast(broadcastDim, targetSize);
    }
    if (rescaleFactor != 1.0) {
        contentAttrBuilder = contentAttrBuilder.rescale(rescaleFactor);
    }
    // RMS kernel interprets gamma using the activation type, so fold gamma to it here.
    if (mlir::cast<mlir::ShapedType>(constOp.getType()).getElementType() != targetElemType) {
        contentAttrBuilder = contentAttrBuilder.convertElemType(targetElemType);
    }
    contentAttrBuilder = contentAttrBuilder.reshape(ShapeRef({targetSize}));
    auto finalContentAttr = contentAttrBuilder.get();

    auto gammaType = mlir::RankedTensorType::get({targetSize}, targetElemType);
    mlir::Value gamma = Const::DeclareOp::create(builder, appendLoc(loc, "gamma"), gammaType, finalContentAttr);
    return gamma;
}

// Returns the size of the single non-one dimension of gamma (or 1 if gamma is fully scalar).
// Fails if gamma has more than one non-one dimension.
mlir::FailureOr<int64_t> getGammaWidth(mlir::Value gamma) {
    auto shape = Shape(getShape(gamma).raw());
    auto nonOneDims = getNonOneDim(shape);
    if (nonOneDims.size() > 1) {
        return mlir::failure();
    }
    return nonOneDims.empty() ? int64_t(1) : shape[nonOneDims.back()];
}

IE::RMSOp createRMSOp(mlir::OpBuilder& builder, mlir::Operation* headOp, mlir::Value gamma, int64_t layerSize,
                      mlir::FloatAttr epsilonAttr, IE::EpsMode epsMode) {
    auto gammaRank = mlir::cast<vpux::NDTypeInterface>(gamma.getType()).getRank();
    if (gammaRank != 1) {
        auto reshapeOp =
                IE::ReshapeOp::create(builder, gamma.getLoc(), gamma,
                                      getIntArrayAttr(headOp->getContext(), SmallVector<int64_t>({layerSize})));
        gamma = reshapeOp;
    }
    auto epsModeAttr = IE::EpsModeAttr::get(headOp->getContext(), epsMode);
    auto rmsOp = IE::RMSOp::create(builder, appendLoc(headOp->getLoc(), "rms"), headOp->getOperand(0), gamma,
                                   epsilonAttr, epsModeAttr);
    return rmsOp;
}

// Match pattern:
// Input -> IE.NormalizeL2 -> IE.Multiply(scale)
// Fuses to IE.RMSOp with gamma = scale, using the *_SUM EpsMode so the kernel computes
// rsqrt(Sum(x^2)+eps) directly instead of dividing by reduceSize.

void fuseNormalizeL2Pattern(IE::NormalizeL2Op normalizeL2Op, mlir::MLIRContext& ctx, vpux::Logger log) {
    const auto epsMode = normalizeL2Op.getEpsMode();
    if (epsMode != IE::EpsMode::ADD && epsMode != IE::EpsMode::MAX) {
        return;
    }

    const auto axes = Const::getConstOrArrAttrValue(normalizeL2Op.getAxes(), normalizeL2Op.getAxesValueAttr());
    if (mlir::failed(axes) || axes->empty()) {
        return;
    }

    // Only fuse NormalizeL2 to RMS when reducing on the innermost dimension
    const auto inputRank = mlir::cast<vpux::NDTypeInterface>(normalizeL2Op.getData().getType()).getRank();
    const int64_t normalizedAxis = axes.value()[0] < 0 ? axes.value()[0] + inputRank : axes.value()[0];
    const bool isInnermost = (axes->size() == 1) && (normalizedAxis == inputRank - 1);
    if (!isInnermost) {
        return;
    }

    auto inputShape = getShape(normalizeL2Op.getData());
    auto reduceSizeResult = getReducedSize(*axes, inputShape);
    if (mlir::failed(reduceSizeResult)) {
        return;
    }
    int64_t reduceSize = reduceSizeResult.value();

    const float epsL2 = static_cast<float>(normalizeL2Op.getEps().convertToDouble());
    const auto epsilonAttr = getFPAttr(&ctx, epsL2);
    const auto sumEpsMode = (epsMode == IE::EpsMode::MAX) ? IE::EpsMode::MAX_SUM : IE::EpsMode::ADD_SUM;

    // NormalizeL2 must have a single use
    if (!normalizeL2Op->hasOneUse()) {
        return;
    }

    mlir::Operation* normalizeL2User = *normalizeL2Op->getUsers().begin();
    auto scaleMultiplyOp = mlir::dyn_cast<IE::MultiplyOp>(normalizeL2User);
    if (scaleMultiplyOp == nullptr) {
        return;
    }

    // Gamma is whichever operand of the Multiply is not the NormalizeL2 output.
    mlir::Value gammaValue = (scaleMultiplyOp.getInput1() == normalizeL2Op.getOutput()) ? scaleMultiplyOp.getInput2()
                                                                                        : scaleMultiplyOp.getInput1();

    mlir::OpBuilder builder(scaleMultiplyOp);
    const auto inputElemType = mlir::cast<vpux::NDTypeInterface>(normalizeL2Op.getData().getType()).getElementType();

    mlir::Value gamma;
    if (Const::DeclareOp scaleConstOp = gammaValue.getDefiningOp<Const::DeclareOp>()) {
        auto gammaResult = broadcastAndRescaleGamma(builder, scaleConstOp, reduceSize, 1.0, inputElemType,
                                                    normalizeL2Op->getLoc());
        if (mlir::failed(gammaResult)) {
            return;
        }
        gamma = gammaResult.value();
    } else {
        // Non-constant gamma: only supported when its width already matches reduceSize, since
        // broadcasting a non-constant value would require inserting a new graph op, which is out
        // of scope here.
        auto gammaShape = Shape(getShape(gammaValue).raw());
        auto nonOneDims = getNonOneDim(gammaShape);
        const bool isTrailingVector =
                gammaShape.size() <= static_cast<size_t>(inputRank) &&
                (nonOneDims.empty() || nonOneDims.back().ind() == static_cast<int64_t>(gammaShape.size() - 1));
        auto gammaWidthResult = getGammaWidth(gammaValue);
        if (mlir::failed(gammaWidthResult) || !isTrailingVector || gammaWidthResult.value() != reduceSize) {
            log.trace("Skipping NormalizeL2-to-RMS fusion: non-constant gamma must be a trailing broadcast vector of "
                      "width {0}",
                      reduceSize);
            return;
        }
        // RMS kernel interprets gamma using the activation type, so reject mismatched non-constant gamma.
        if (mlir::cast<vpux::NDTypeInterface>(gammaValue.getType()).getElementType() != inputElemType) {
            log.trace("Skipping NormalizeL2-to-RMS fusion: non-constant gamma element type does not match input");
            return;
        }
        gamma = gammaValue;
    }

    auto rmsOp = createRMSOp(builder, normalizeL2Op, gamma, reduceSize, epsilonAttr, sumEpsMode);
    scaleMultiplyOp->replaceAllUsesWith(rmsOp);
}

//
// safeRunOnFunc
//

void FuseNormalizeL2ToRMSPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    func->walk([&](IE::NormalizeL2Op normalizeL2Op) {
        fuseNormalizeL2Pattern(normalizeL2Op, ctx, _log);
    });
}

}  // namespace

//
// createFuseNormalizeL2ToRMSPass
//

std::unique_ptr<mlir::Pass> vpux::IE::createFuseNormalizeL2ToRMSPass(Logger log) {
    return std::make_unique<FuseNormalizeL2ToRMSPass>(log);
}
