//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace vpux::VPU {

// Legacy weightsTable and split scale/bias table operands are mutually exclusive.
struct TileWithEltwiseWeightTables final {
    mlir::Value weightsTable;
    mlir::Value scale;
    mlir::Value bias;
};

//
// NCEEltwiseRewriterRewrite
//
// Common Tile + NCE.Eltwise rewriter; derived strategies select the HW-specific WT format.
//
class ConvertTileWithEltwiseToNCEPoolRewriter : public mlir::OpRewritePattern<VPU::TileOp> {
public:
    ConvertTileWithEltwiseToNCEPoolRewriter(mlir::MLIRContext* ctx, Logger log);

    mlir::LogicalResult matchAndRewrite(VPU::TileOp origOp, mlir::PatternRewriter& rewriter) const final;

protected:
    virtual TileWithEltwiseWeightTables createWeightTableOperands(mlir::Location loc, mlir::Value scale,
                                                                  mlir::Value bias, int64_t weightsTableC,
                                                                  mlir::PatternRewriter& rewriter) const = 0;

protected:
    Logger _log;
};

}  // namespace vpux::VPU
