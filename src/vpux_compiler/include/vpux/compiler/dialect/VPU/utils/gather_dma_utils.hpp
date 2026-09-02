//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement_fwd.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/gather_dma_constants.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>

namespace vpux::VPU {

// Gather DMA only support dims before axis equal to 1, checkDimsBeforeAxis is used when we try to do some conversion
// to reshape dims before axis to 1.
bool isLegalConvertToGatherDMA(VPU::GatherOp op, bool isElementTile, bool isIndicesTile, vpux::Logger log,
                               bool checkDimsBeforeAxis = true);

Shape getSupportedNTilesOnDimforGather(ArrayRef<int64_t> tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                       Logger log);

Shape getSupportedNTilesOnDimforGatherElements(DimArrRef tileDimOrder, mlir::Operation* baseOp, TilingMode tilingMode,
                                               Logger log);

bool isOutermostGatherDMAWithLowBit(VPU::GatherDMAOp origOp);

}  // namespace vpux::VPU
