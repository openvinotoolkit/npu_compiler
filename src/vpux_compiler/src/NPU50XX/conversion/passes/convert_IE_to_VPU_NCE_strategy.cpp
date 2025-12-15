//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/conversion/passes/convert_IE_to_VPU_NCE_strategy.hpp"
#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/IE/utils/reduce_infer.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"

namespace vpux::arch50xx {
template <typename ConcreteOp>
class ReduceToNCE final : public mlir::OpRewritePattern<ConcreteOp> {
public:
    ReduceToNCE<ConcreteOp>(mlir::MLIRContext* ctx, VPU::ReduceType opType, config::ArchKind arch, Logger log)
            : mlir::OpRewritePattern<ConcreteOp>(ctx), _opType(opType), _arch(arch), _log(log) {
    }

    mlir::LogicalResult matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    VPU::ReduceType _opType;
    config::ArchKind _arch;
    Logger _log;
};

template <typename ConcreteOp>
mlir::LogicalResult ReduceToNCE<ConcreteOp>::matchAndRewrite(ConcreteOp origOp, mlir::PatternRewriter& rewriter) const {
    auto axes = getIntArrayAttr(this->getContext(), IE::extractAxes(origOp->getLoc(), origOp));
    auto ppeAttr = VPU::PpeVersionConfig::retrievePPEAttribute(origOp);
    auto nceOp = rewriter.create<VPU::NCEReduceOp>(origOp->getLoc(), origOp.getType(), origOp.getInput(), axes, ppeAttr,
                                                   VPU::ReduceTypeAttr::get(this->getContext(), _opType),
                                                   /*multiClusterStrategy=*/nullptr, origOp.getOutputPaddingAttr(),
                                                   origOp.getInputPaddingAttr());

    rewriter.replaceOp(origOp, nceOp.getOutput());
    return mlir::success();
}

void ConvertIEToVPUNCEStrategy::addTargets(mlir::ConversionTarget& target, LogCb logCb) const {
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<vpux::IE::IEDialect>();
    target.addLegalDialect<vpux::VPU::VPUDialect>();
    target.addDynamicallyLegalOp<IE::MatMulOp>([logCb](IE::MatMulOp op) {
        // Layout correction and transformation to 5D is done during lowering so layout check is disabled.
        // Expected layout is intentionally NCHW, ensured by AdjustLayouts Pass.
        return !VPU::NCEMatMulOp::isSupported(op, logCb, /* checkLayout = */ false,
                                              /* checkChannelAlignment = */ true) ||
               VPU::MatMulOp::isSupported(op);
    });

    target.addDynamicallyLegalOp<IE::ConvolutionOp>([logCb](IE::ConvolutionOp op) {
        return !VPU::NCEConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                   /*checkChannelAlignment=*/true) &&
               !VPU::NCECompressConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                           /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::GroupConvolutionOp>([logCb](IE::GroupConvolutionOp op) {
        if (VPU::isDilatedGroupConv(op)) {
            return true;
        }

        return !VPU::NCEDepthConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                        /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::MaxPoolOp>([logCb](IE::MaxPoolOp op) {
        return !VPU::NCEMaxPoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });
    target.addDynamicallyLegalOp<IE::AvgPoolOp>([logCb](IE::AvgPoolOp op) {
        return !VPU::NCEAveragePoolOp::isSupported(op, logCb, /*checkLayout=*/true,
                                                   /*checkChannelAlignment=*/true);
    });

    target.addDynamicallyLegalOp<IE::PermuteQuantizeOp>([logCb](IE::PermuteQuantizeOp op) {
        return !VPU::NCEPermuteOp::isSupported(op, logCb, /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });

    target.addDynamicallyLegalOp<IE::AddOp, IE::SubtractOp, IE::MultiplyOp>([logCb](mlir::Operation* op) {
        const bool allowDifferentScales = true;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb,
                                               /*checkLayout=*/true,
                                               /*checkChannelAlignment=*/true);
    });

    target.addDynamicallyLegalOp<IE::ReduceMeanOp, IE::ReduceSumOp>([logCb](mlir::Operation* op) {
        return !VPU::NCEReduceOp::isSupported(op, logCb, /*checkLayout=*/true,
                                              /*checkChannelAlignment=*/true);
    });
}

void ConvertIEToVPUNCEStrategy::addPatterns(mlir::RewritePatternSet& patterns) const {
    auto ctx = patterns.getContext();

    patterns.add<ConvToNCE>(ctx, _arch, _log);
    patterns.add<DepthConvToNCE>(ctx, _arch, _log);
    patterns.add<MaxPoolToNCE>(ctx, _log);
    patterns.add<AveragePoolToNCE>(ctx, _log);
    patterns.add<PermuteQuantizeToNCEPermute>(ctx, _log);
    patterns.add<MatMulToNCE>(ctx, _arch, _log);

    patterns.add<EltwiseToNCE<IE::AddOp>>(ctx, VPU::EltwiseType::ADD, _log);
    patterns.add<EltwiseToNCE<IE::SubtractOp>>(ctx, VPU::EltwiseType::SUBTRACT, _log);
    patterns.add<EltwiseToNCE<IE::MultiplyOp>>(ctx, VPU::EltwiseType::MULTIPLY, _log);
    patterns.add<arch50xx::ReduceToNCE<IE::ReduceMeanOp>>(ctx, VPU::ReduceType::MEAN, _arch, _log);
    patterns.add<arch50xx::ReduceToNCE<IE::ReduceSumOp>>(ctx, VPU::ReduceType::SUM, _arch, _log);
}
}  // namespace vpux::arch50xx
