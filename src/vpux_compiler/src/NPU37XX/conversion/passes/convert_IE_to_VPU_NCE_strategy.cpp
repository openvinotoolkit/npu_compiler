//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/conversion/passes/convert_IE_to_VPU_NCE_strategy.hpp"
#include "vpux/compiler/conversion/passes/IE2VPU/convert_IE_to_VPU_NCE.hpp"

#include "vpux/compiler/dialect/IE/IR/dialect.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/specialized.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Math/IR/Math.h>

namespace vpux::arch37xx {
void ConvertIEToVPUNCEStrategy::addTargets(mlir::ConversionTarget& target, LogCb logCb) const {
    target.addLegalDialect<Const::ConstDialect>();
    target.addLegalDialect<IE::IEDialect>();
    target.addLegalDialect<VPU::VPUDialect>();
    target.addLegalDialect<mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::math::MathDialect>();

    target.addDynamicallyLegalOp<IE::MatMulOp>([logCb](IE::MatMulOp op) {
        // Layout correction and transformation to 5D is done during lowering so layout check is disabled.
        // Expected layout is intentionally NCHW, ensured by AdjustLayouts Pass.
        return !VPU::NCEMatMulOp::isSupported(op, logCb, /* checkLayout = */ false,
                                              /* checkChannelAlignment = */ true) ||
               VPU::MatMulOp::isSupported(op);
    });

    target.addDynamicallyLegalOp<IE::ConvolutionOp>([logCb](IE::ConvolutionOp op) {
        return !VPU::NCEConvolutionOp::isSupported(op, logCb, /*checkLayout=*/true, /*checkChannelAlignment=*/true) &&
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

    target.addDynamicallyLegalOp<IE::AddOp>([logCb](IE::AddOp op) {
        const bool allowDifferentScales = true;
        const bool allowDifferentZp = true;

        return !VPU::NCEEltwiseOp::isSupported(op, allowDifferentScales, allowDifferentZp, logCb, /*checkLayout=*/true,
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
}
}  // namespace vpux::arch37xx
