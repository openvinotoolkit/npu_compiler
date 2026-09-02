//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/tiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_fwd.hpp"

namespace vpux::VPU {

OutputTiling DetectionOutputSortOpOutputTiling(const vpux::TileInfo& firstOutputTile);
InputTiling DetectionOutputSortOpInputTiling(const vpux::TileInfo& firstOutputTile, int numShaves);
InputTiling DetectionOutputSortOpInputTilingOnShave(VPUIP::SwKernelOp swKernelOp, const vpux::TileInfo& outputTile,
                                                    int tileId, int tileCount, Logger log);

OutputTiling GRUSequenceOutputTiling(const vpux::TileInfo& firstOutputTile);
OutputTiling logSoftmaxTopKOutputTiling(const vpux::TileInfo& firstOutputTile, int64_t axis);
OutputTiling lstmSequenceOutputTiling(const vpux::TileInfo& firstOutputTile);
OutputTiling lstmDpuOutputTiling(const vpux::TileInfo& firstOutputTile);
OutputTiling DynamicQuantizeOutputTiling(const vpux::TileInfo& firstOutputTile, ShapeRef scaleShape, ShapeRef zpShape);

OutputTiling FlashSDPAOpOutputTiling(const vpux::TileInfo& firstOutputTile);
InputTiling FlashSDPAOpInputTiling(const vpux::TileInfo& firstOutputTile, int64_t qHeads, ShapeRef keyShape,
                                   std::optional<ShapeRef> attentionMaskShape, ShapeRef auxBufferShape,
                                   ShapeRef dpuDescriptorBufferShape, ShapeRef weightsTable0Shape,
                                   ShapeRef weightsTable1Shape);
}  // namespace vpux::VPU
