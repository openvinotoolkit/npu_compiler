//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace vpux {
#define GEN_PASS_DECL_ADJUSTDYNAMICOPSBEFOREBUFFERIZATION
#define GEN_PASS_DEF_ADJUSTDYNAMICOPSBEFOREBUFFERIZATION
#include "vpux/compiler/conversion/passes.hpp.inc"
}  // namespace vpux

using namespace vpux;

namespace {

//
// UnsqueezeRewrite
//

class UnsqueezeRewrite final : public mlir::OpRewritePattern<VPU::UnsqueezeOp> {
public:
    UnsqueezeRewrite(mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<VPU::UnsqueezeOp>(ctx), _log(std::move(log)) {
    }

public:
    mlir::LogicalResult matchAndRewrite(VPU::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult UnsqueezeRewrite::matchAndRewrite(VPU::UnsqueezeOp origOp, mlir::PatternRewriter& rewriter) const {
    _log.trace("[{0}] Rewriting '{1}' at '{2}'", getDebugName(), origOp->getName(), origOp->getLoc());

    const auto ctx = origOp->getContext();

    const auto output = origOp.getOutput();
    const auto outputShape = getShape(output);

    const auto boundedTensorType = mlir::cast<Core::BoundedTensorType>(output.getType());

    SmallVector<int32_t> zeroedDynamicDimsShape;
    zeroedDynamicDimsShape.reserve(outputShape.size());
    std::transform(outputShape.begin(), outputShape.end(), std::back_inserter(zeroedDynamicDimsShape), [](int64_t dim) {
        return (dim == mlir::ShapedType::kDynamic) ? 0 : dim;
    });

    const auto outputRank = checked_cast<int64_t>(outputShape.size());
    const auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto shapeType = mlir::RankedTensorType::get({outputRank}, si32Type);
    const auto shapeTensor =
            Const::createConst(rewriter, origOp->getLoc(), shapeType, ArrayRef(zeroedDynamicDimsShape));

    const auto outputShapeAttr = getIntArrayAttr(ctx, outputShape);
    const auto outputBoundsAttr = getIntArrayAttr(ctx, boundedTensorType.getBounds());
    rewriter.replaceOpWithNewOp<VPU::DynamicReshapeOp>(origOp, origOp.getType(), origOp.getInput(), shapeTensor,
                                                       outputShapeAttr, outputBoundsAttr, /*only_set_shape*/ false,
                                                       VPU::BoundsRepresentation::BOUNDS);
    return mlir::success();
}

//
// AdjustDynamicOpsBeforeBufferizationPass
//

class AdjustDynamicOpsBeforeBufferizationPass final :
        public impl::AdjustDynamicOpsBeforeBufferizationBase<AdjustDynamicOpsBeforeBufferizationPass> {
private:
    void safeRunOnModule() final;
};

void AdjustDynamicOpsBeforeBufferizationPass::safeRunOnModule() {
    auto& ctx = getContext();

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<Const::ConstDialect>();
    target.addDynamicallyLegalOp<VPU::UnsqueezeOp>(std::not_fn(IE::hasDynamicTensors));
    target.addLegalOp<VPU::DynamicReshapeOp>();

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<UnsqueezeRewrite>(&ctx, _log);

    auto func = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createAdjustDynamicOpsBeforeBufferizationPass
//

std::unique_ptr<mlir::Pass> vpux::createAdjustDynamicOpsBeforeBufferizationPass() {
    return std::make_unique<AdjustDynamicOpsBeforeBufferizationPass>();
}
