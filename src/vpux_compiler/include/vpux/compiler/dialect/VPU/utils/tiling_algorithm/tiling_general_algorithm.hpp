//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_alg_interface.hpp"

namespace vpux {
namespace VPU {
//
// TilingGeneralAlgorithm
//

// Tiling algorithm with Slice-Concat approach
class TilingGeneralAlgorithm final : public ITilingAlgorithm {
public:
    mlir::LogicalResult applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder, Logger log) override;

    SmallVector<mlir::Operation*> applySCFTilingAndFusion(mlir::Operation* operation, mlir::RewriterBase& builder,
                                                          const MergeConfiguration& mergeConfig, Logger log) override;
};
}  // namespace VPU
}  // namespace vpux
