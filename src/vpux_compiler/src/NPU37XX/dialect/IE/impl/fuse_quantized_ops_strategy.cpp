//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_quantized_ops.hpp"
#include "vpux/compiler/dialect/IE/utils/fake_quantize_utils.hpp"
#include "vpux/compiler/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/VPU/utils/conv_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/nce_invariant.hpp"

namespace vpux::IE::arch37xx {

//
// FuseWithDepth2Space
//

//
//       [input]
//          |
//     (dequantize)
//          |
//        (DepthToSpace)
//          |
//       [output]
//          |
//      (quantize)
//

class FuseWithDepth2Space final : public mlir::OpRewritePattern<IE::QuantizeOp> {
public:
    FuseWithDepth2Space(mlir::MLIRContext* ctx, Logger log): mlir::OpRewritePattern<IE::QuantizeOp>(ctx), _log(log) {
        setDebugName("FuseWithDepth2Space");
    }

public:
    mlir::LogicalResult matchAndRewrite(IE::QuantizeOp quantizeOp, mlir::PatternRewriter& rewriter) const final;

private:
    Logger _log;
};

mlir::LogicalResult FuseWithDepth2Space::matchAndRewrite(IE::QuantizeOp quantizeOp,
                                                         mlir::PatternRewriter& rewriter) const {
    if (isPerAxisQuant(quantizeOp.getOutput())) {
        return mlir::failure();
    }

    auto depth2SpaceOp = quantizeOp.getInput().getDefiningOp<IE::DepthToSpaceOp>();
    if (depth2SpaceOp == nullptr) {
        return mlir::failure();
    }

    if (!areAllUsersQuantized(depth2SpaceOp)) {
        return mlir::failure();
    }

    auto inputDequantizeOp = depth2SpaceOp.getInput().getDefiningOp<IE::DequantizeOp>();
    if (inputDequantizeOp == nullptr) {
        return mlir::failure();
    }

    if (isPerAxisQuant(inputDequantizeOp.getInput())) {
        return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<IE::DepthToSpaceOp>(quantizeOp, quantizeOp.getType(), inputDequantizeOp.getInput(),
                                                    depth2SpaceOp.getBlockSizeAttr(), depth2SpaceOp.getModeAttr())
            ->setLoc(depth2SpaceOp->getLoc());

    return mlir::success();
}

//
// FuseQuantizedOpsStrategy
//

void FuseQuantizedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    const auto checkAddInputTypes = [&](mlir::Type input1Type, mlir::Type input2Type,
                                        VPU::EltwiseType eltwiseType) -> mlir::LogicalResult {
        auto dequantElemIn1Type = mlir::cast<mlir::quant::UniformQuantizedType>(input1Type);
        auto dequantElemIn2Type = mlir::cast<mlir::quant::UniformQuantizedType>(input2Type);

        // Perform check for input types. AddOp supports quantization with different zp, but not different scales.
        if (dequantElemIn1Type.getExpressedType() != dequantElemIn2Type.getExpressedType() ||
            dequantElemIn1Type.getStorageType() != dequantElemIn2Type.getStorageType() ||
            dequantElemIn1Type.isSigned() != dequantElemIn2Type.isSigned()) {
            return mlir::failure();
        }

        if (!isSupportedEltwiseQuantization(dequantElemIn1Type, dequantElemIn2Type, /*allowDifferentScales=*/true,
                                            /*allowDifferentZp=*/true, eltwiseType)) {
            return mlir::failure();
        }

        return mlir::success();
    };

    patterns.add<FuseWithConv>(ctx, checkPostOp, false, log);
    patterns.add<FuseWithGroupConv>(ctx, checkPostOp, true, log);
    patterns.add<FuseWithEltwiseConverter<IE::AddOp>>(ctx, checkPostOp, checkAddInputTypes, VPU::EltwiseType::ADD,
                                                      false, log);
    patterns.add<FuseWithSlice>(ctx, log);
    patterns.add<FuseWithMaxPool>(ctx, false, log);
    patterns.add<FuseWithTile>(ctx, log);
    patterns.add<FuseWithAveragePool>(ctx, false, log);
    patterns.add<FuseWithConcat>(ctx, log);
    patterns.add<FuseWithDepth2Space>(ctx, log);
    patterns.add<FuseWithMatMul>(ctx, log);
    patterns.add<FuseWithPostOp>(ctx, log);
    if (_seOpsEnabled) {
        patterns.add<FuseWithInterpolate>(ctx, log);
        patterns.add<FuseWithTransposedConv>(ctx, checkPostOp, false, log);
    }

    // TODO: optimize for SEP Pad & Roll
    VPUX_UNUSED(_seExperimentalOpsEnabled);
}

}  // namespace vpux::IE::arch37xx
