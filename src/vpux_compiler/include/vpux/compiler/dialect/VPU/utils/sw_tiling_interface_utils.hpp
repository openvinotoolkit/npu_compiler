//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/utils/VPU/tile_utils.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>

using namespace vpux;

template <class ConcreteModel, class MainOpType>
class SwLayerTilingInfoOpModelBase : public VPU::TilingInfoOpInterface::ExternalModel<ConcreteModel, MainOpType> {
public:
    bool isSupportedTiling(mlir::Operation* origOp, const OutputTiling& tiles, TilingMode tilingMode,
                           Logger log) const {
        switch (tilingMode) {
        case vpux::TilingMode::ISOLATED:
            return vpux::VPU::isSupportedIsolatedTilingSwLayer(origOp, tiles, log);
        case vpux::TilingMode::PIPELINING:
            return vpux::VPU::isSupportedPipeliningTilingSwLayer(origOp, tiles, log);
        case vpux::TilingMode::PREFETCHING:
            return false;
        default:
            VPUX_THROW("Unknown tiling mode: '{0}'.", getTilingModeStr(tilingMode));
        }
    }

    bool isSupportedTilingStrategy(mlir::Operation* origOp, const vpux::Shape& strategy, TilingMode tilingMode,
                                   Logger log) const {
        return vpux::VPU::isSupportedTilingStrategyImpl(origOp, strategy, tilingMode, log);
    }
};
