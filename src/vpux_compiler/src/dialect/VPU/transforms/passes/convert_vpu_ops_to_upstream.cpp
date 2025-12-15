//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_CONVERTVPUOPSTOUPSTREAMOPS
#define GEN_PASS_DEF_CONVERTVPUOPSTOUPSTREAMOPS
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {
class SliceOpConversion : public mlir::OpConversionPattern<vpux::VPU::SliceOp> {
    using mlir::OpConversionPattern<vpux::VPU::SliceOp>::OpConversionPattern;

    mlir::LogicalResult matchAndRewrite(vpux::VPU::SliceOp op, OpAdaptor adaptor,
                                        mlir::ConversionPatternRewriter& rewriter) const override;
};
}  // namespace

mlir::LogicalResult SliceOpConversion::matchAndRewrite(vpux::VPU::SliceOp op, OpAdaptor adaptor,
                                                       mlir::ConversionPatternRewriter& rewriter) const {
    if (mlir::isa<vpux::VPU::SparseTensorType>(adaptor.getInput().getType())) {
        return mlir::failure();
    }
    const auto sliceOffsets =
            mlir::getAsIndexOpFoldResult(rewriter.getContext(), parseIntArrayAttr<int64_t>(adaptor.getStaticOffsets()));
    SmallVector<mlir::OpFoldResult> defaultStrides(sliceOffsets.size(), rewriter.getIndexAttr(1));

    mlir::ReifiedRankedShapedTypeDims reifiedShapes;
    if (mlir::failed(reifyResultShapes(rewriter, op, reifiedShapes))) {
        return mlir::failure();
    }
    auto newSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(op.getLoc(), adaptor.getInput(), sliceOffsets,
                                                                  reifiedShapes[0], defaultStrides);
    newSlice.getResult().setType(mlir::cast<mlir::RankedTensorType>(op.getOutput().getType()));
    rewriter.replaceOp(op, newSlice.getResult());

    return mlir::success();
}

namespace {

//
// ConvertVPUOpsToUpstreamOpsPass
//
class ConvertVPUOpsToUpstreamOpsPass :
        public VPU::impl::ConvertVPUOpsToUpstreamOpsBase<ConvertVPUOpsToUpstreamOpsPass> {
public:
    explicit ConvertVPUOpsToUpstreamOpsPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }
    void safeRunOnModule() final;
};

void ConvertVPUOpsToUpstreamOpsPass::safeRunOnModule() {
    auto& ctx = getContext();
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::func::FuncOp, mlir::func::ReturnOp, mlir::func::CallOp>();

    // Setup conversion patterns.
    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<SliceOpConversion>(patterns.getContext());

    // Apply conversion.
    auto module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

}  // namespace

//
// createConvertVPUOpsToUpstreamOpsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createConvertVPUOpsToUpstreamOpsPass(Logger log) {
    return std::make_unique<ConvertVPUOpsToUpstreamOpsPass>(log);
}
