//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mlir/IR/Operation.h>
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"

namespace vpux::VPU {
class GatherOp;
}

namespace vpux::VPU {

bool isLegalConvertToGatherDMA(VPU::GatherOp op, bool isElementTile, bool isIndicesTile, vpux::Logger log);

Shape getSupportedNTilesOnDimforGather(ArrayRef<int64_t> tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                       Logger log);

Shape getSupportedNTilesOnDimforGatherElements(DimArrRef tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                               Logger log);

}  // namespace vpux::VPU
