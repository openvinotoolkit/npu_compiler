//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>

#include <functional>
#include <utility>
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/shape_manipulation.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/utils/core/checked_cast.hpp"

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

    const auto dynamicDimsMaskType = mlir::cast<Core::DynamicDimsMaskTensorType>(output.getType());
    const auto dynamicDimsMask = dynamicDimsMaskType.getDynamicDimsMask();

    auto dynamicShape = SmallVector<int64_t>();
    auto zeroedDynamicDimsShape = SmallVector<int32_t>();
    for (auto [dim, mask] : zip(outputShape, dynamicDimsMask.raw())) {
        auto dynamicDim = (mask == 0) ? dim : mlir::ShapedType::kDynamic;
        dynamicShape.push_back(dynamicDim);

        auto zeroedDim = (mask == 0) ? checked_cast<int32_t>(dim) : 0;
        zeroedDynamicDimsShape.push_back(zeroedDim);
    }

    const auto outputRank = checked_cast<int64_t>(outputShape.size());
    const auto si32Type = mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed);
    const auto shapeType = mlir::RankedTensorType::get({outputRank}, si32Type);
    const auto shapeTensor =
            Const::createConst(rewriter, origOp->getLoc(), shapeType, ArrayRef(zeroedDynamicDimsShape));

    const auto outputShapeAttr = getIntArrayAttr(ctx, dynamicShape);
    const auto outputBoundsAttr = getIntArrayAttr(ctx, outputShape.raw());
    rewriter.replaceOpWithNewOp<VPU::DynamicReshapeOp>(origOp, origOp.getType(), origOp.getInput(), shapeTensor,
                                                       outputShapeAttr, outputBoundsAttr, /*only_set_shape*/ false,
                                                       VPU::BoundsRepresentation::DYNAMIC_DIMS_MASK);
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
