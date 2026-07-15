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

Const::DeclareOp getMultiplyConstOperand(mlir::Operation* op) {
    auto const1 = op->getOperand(0).getDefiningOp<Const::DeclareOp>();
    auto const2 = op->getOperand(1).getDefiningOp<Const::DeclareOp>();
    return const1 ? const1 : const2;
}

mlir::FailureOr<int64_t> getReducedSize(ArrayRef<int64_t> reduceAxes, vpux::ShapeRef reduceInputShape) {
    int64_t reduceSize = 1;
    for (const auto& axis : reduceAxes | indexed) {
        if (axis.value() != (int64_t)(reduceInputShape.size() - reduceAxes.size() + axis.index())) {
            return mlir::failure();
        }
        reduceSize *= reduceInputShape[Dim(axis.value())];
    }
    return reduceSize;
}

mlir::FailureOr<mlir::Value> broadcastAndRescaleGamma(mlir::OpBuilder& builder, Const::DeclareOp constOp,
                                                      int64_t targetSize, double rescaleFactor, mlir::Location loc) {
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
    contentAttrBuilder = contentAttrBuilder.reshape(ShapeRef({targetSize}));
    auto finalContentAttr = contentAttrBuilder.get();

    auto gammaType =
            mlir::RankedTensorType::get({targetSize}, mlir::cast<mlir::ShapedType>(constOp.getType()).getElementType());
    mlir::Value gamma = builder.create<Const::DeclareOp>(appendLoc(loc, "gamma"), gammaType, finalContentAttr);
    return gamma;
}

IE::RMSOp createRMSOp(mlir::OpBuilder& builder, mlir::Operation* headOp, mlir::Value gamma, int64_t layerSize,
                      mlir::FloatAttr epsilonAttr) {
    auto gammaRank = mlir::cast<vpux::NDTypeInterface>(gamma.getType()).getRank();
    if (gammaRank != 1) {
        auto reshapeOp = builder.create<IE::ReshapeOp>(
                gamma.getLoc(), gamma, getIntArrayAttr(headOp->getContext(), SmallVector<int64_t>({layerSize})));
        gamma = reshapeOp;
    }
    auto rmsOp =
            builder.create<IE::RMSOp>(appendLoc(headOp->getLoc(), "rms"), headOp->getOperand(0), gamma, epsilonAttr);
    return rmsOp;
}

// Match pattern:
// Input -> IE.NormalizeL2 -> IE.Multiply(scale)
// Fuses to IE.RMSOp with gamma = scale / sqrt(reduceSize)

void fuseNormalizeL2Pattern(IE::NormalizeL2Op normalizeL2Op, mlir::MLIRContext& ctx, vpux::Logger /*log*/) {
    if (!normalizeL2Op.getAxesValue()) {
        return;
    }

    const auto epsMode = normalizeL2Op.getEpsMode();
    if (epsMode != IE::EpsMode::ADD) {
        return;
    }

    const auto axes = parseIntArrayAttr<int64_t>(normalizeL2Op.getAxesValueAttr());
    if (axes.empty()) {
        return;
    }

    // Only fuse NormalizeL2 to RMS when reducing on the innermost dimension
    const auto inputRank = mlir::cast<vpux::NDTypeInterface>(normalizeL2Op.getData().getType()).getRank();
    const int64_t normalizedAxis = axes[0] < 0 ? axes[0] + inputRank : axes[0];
    const bool isInnermost = (axes.size() == 1) && (normalizedAxis == inputRank - 1);
    if (!isInnermost) {
        return;
    }

    auto inputShape = getShape(normalizeL2Op.getData());
    auto reduceSizeResult = getReducedSize(axes, inputShape);
    if (mlir::failed(reduceSizeResult)) {
        return;
    }
    int64_t reduceSize = reduceSizeResult.value();

    const float epsL2 = static_cast<float>(normalizeL2Op.getEps().convertToDouble());
    // NormalizeL2 uses ReduceSum, RMS uses ReduceMean; scale epsilon accordingly
    const auto epsilonAttr = getFPAttr(&ctx, epsL2 / static_cast<float>(reduceSize));
    const float gammaScale = sqrtf(static_cast<float>(reduceSize));

    // NormalizeL2 must have a single use
    if (!normalizeL2Op->hasOneUse()) {
        return;
    }

    mlir::Operation* normalizeL2User = *normalizeL2Op->getUsers().begin();
    auto scaleMultiplyOp = mlir::dyn_cast<IE::MultiplyOp>(normalizeL2User);
    if (scaleMultiplyOp == nullptr) {
        return;
    }

    Const::DeclareOp scaleConstOp = getMultiplyConstOperand(scaleMultiplyOp);
    if (scaleConstOp == nullptr) {
        return;
    }

    mlir::OpBuilder builder(scaleMultiplyOp);

    auto gammaResult = broadcastAndRescaleGamma(builder, scaleConstOp, reduceSize,
                                                1.0 / static_cast<double>(gammaScale), normalizeL2Op->getLoc());
    if (mlir::failed(gammaResult)) {
        return;
    }
    mlir::Value gamma = gammaResult.value();

    auto rmsOp = createRMSOp(builder, normalizeL2Op, gamma, reduceSize, epsilonAttr);
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
