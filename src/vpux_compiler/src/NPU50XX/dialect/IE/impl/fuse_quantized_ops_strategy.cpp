//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/IE/impl/fuse_quantized_ops_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/IE/utils/quantization.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/reduce.hpp"
#include "vpux/compiler/dialect/IE/interfaces/common_rewriters/fuse_quantized_ops.hpp"
#include "vpux/utils/core/numeric.hpp"

namespace vpux::IE::arch50xx {

//
// FuseQuantizedOpsStrategy
//

void FuseQuantizedOpsStrategy::addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const {
    auto ctx = patterns.getContext();

    const auto checkInputTypes = [&](mlir::Type input1Type, mlir::Type input2Type,
                                     VPU::EltwiseType eltwiseType) -> mlir::LogicalResult {
        auto dequantElemIn1Type = mlir::cast<mlir::quant::UniformQuantizedType>(input1Type);
        auto dequantElemIn2Type = mlir::cast<mlir::quant::UniformQuantizedType>(input2Type);
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
    patterns.add<FuseWithEltwiseConverter<IE::AddOp>>(ctx, checkPostOp, checkInputTypes, VPU::EltwiseType::ADD, false,
                                                      log);
    patterns.add<FuseWithEltwiseConverter<IE::SubtractOp>>(ctx, checkPostOp, checkInputTypes,
                                                           VPU::EltwiseType::SUBTRACT, false, log);
    // #E157147: Disable fuse quantized for multiply. It will be enabled once it is optimal.
    patterns.add<FuseWithSlice>(ctx, log);
    patterns.add<FuseWithMaxPool>(ctx, true, log);
    patterns.add<FuseWithTile>(ctx, log);
    patterns.add<FuseWithReduce<IE::ReduceMeanOp>>(ctx, log);
    patterns.add<FuseWithReduce<IE::ReduceSumOp>>(ctx, log);
    patterns.add<FuseWithAveragePool>(ctx, false, log);
    patterns.add<FuseWithConcat>(ctx, log);
    patterns.add<FuseWithMatMul>(ctx, log);
    patterns.add<FuseWithPostOp>(ctx, log);
    if (_seOpsEnabled) {
        patterns.add<FuseWithInterpolate>(ctx, log);
        patterns.add<FuseWithTransposedConv>(ctx, checkPostOp, false, log);
    }

    // TODO: optimize for SEP Pad & Roll
    VPUX_UNUSED(_seExperimentalOpsEnabled);
}

}  // namespace vpux::IE::arch50xx
