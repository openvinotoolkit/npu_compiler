//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_general_algorithm.hpp"
#include "vpux/compiler/dialect/VPU/utils/generate_tiling.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/precomputed_strategy_table_cache.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;
using namespace VPU;

mlir::LogicalResult TilingGeneralAlgorithm::applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                        Logger log) {
    if (!operation->hasAttr(tilingStrategy)) {
        return mlir::failure();
    }
    const auto strategy =
            Shape(parseIntArrayAttr<int64_t>(mlir::cast<mlir::ArrayAttr>(operation->getAttr(tilingStrategy))));

    auto tilingBuilder = mlir::dyn_cast<VPU::TilingBuilderOpInterface>(operation);
    VPUX_THROW_WHEN(tilingBuilder == nullptr, "Operation '{0}' doesn't implement TilingInfoOpInterface",
                    operation->getName());

    const bool efficientWorkloadAlign = operation->hasAttr(VPU::pinnedStrategy) ? true : false;
    const auto tiles = fillDividedTiles(operation, strategy, getShape(operation->getResult(0)), efficientWorkloadAlign);

    if (mlir::failed(tiles)) {
        return mlir::failure();
    }
    operation->removeAttr(tilingStrategy);
    return VPU::applyTileStrategy(tilingBuilder, tiles.value(), builder, log);
}

SmallVector<mlir::Operation*> TilingGeneralAlgorithm::applySCFTilingAndFusion(mlir::Operation*, mlir::RewriterBase&,
                                                                              const MergeConfiguration&, Logger) {
    // TODO E-172818 move VF general algorithm here
    return {};
}
