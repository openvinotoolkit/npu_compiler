//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/impl/convert_tile_with_eltwise_to_nce_pool_strategy.hpp"

#include "vpux/compiler/dialect/VPU/interfaces/common_rewriters/convert_tile_with_eltwise_to_nce_pool.hpp"

using namespace vpux;

namespace vpux::VPU::arch60xx {

namespace {

class ConvertTileWithEltwiseToNCEPool final : public VPU::ConvertTileWithEltwiseToNCEPoolRewriter {
public:
    ConvertTileWithEltwiseToNCEPool(mlir::MLIRContext* ctx, Logger log)
            : VPU::ConvertTileWithEltwiseToNCEPoolRewriter(ctx, log) {
    }

private:
    VPU::TileWithEltwiseWeightTables createWeightTableOperands(mlir::Location, mlir::Value scale, mlir::Value bias,
                                                               int64_t, mlir::PatternRewriter&) const final {
        return {/*weightsTable=*/nullptr, scale, bias};
    }
};

}  // namespace

void ConvertTileWithEltwiseToNCEPoolStrategy::addPatterns(SmallVector<mlir::RewritePatternSet>& patterns,
                                                          Logger& log) const {
    mlir::RewritePatternSet patternSet(_ctx);
    patternSet.add<ConvertTileWithEltwiseToNCEPool>(_ctx, log);
    patterns.push_back(std::move(patternSet));
}

}  // namespace vpux::VPU::arch60xx
