//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"

#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

#include "vpux/utils/core/small_vector.hpp"

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"

namespace vpux {
namespace VPU {
//
// ITilingAlgorithm
//

// The interface of tiling algorithm
class ITilingAlgorithm {
public:
    virtual ~ITilingAlgorithm() = default;

    virtual mlir::LogicalResult applyTiling(mlir::Operation* operation, mlir::RewriterBase& builder, Logger log) = 0;

    virtual SmallVector<mlir::Operation*> applySCFTilingAndFusion(mlir::Operation* operation,
                                                                  mlir::RewriterBase& builder,
                                                                  const MergeConfiguration& mergeConfig,
                                                                  Logger log) = 0;
};
}  // namespace VPU
}  // namespace vpux
