//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_alg_interface.hpp"

namespace vpux {
namespace VPU {
//
// TilingContext
//

// class which stores information of tiling algorithm
// which is going to be applied and runs tiling based on it
class TilingContext final {
public:
    TilingContext(mlir::Operation* operation);

    void setTiling(std::unique_ptr<ITilingAlgorithm> tilingAlgorithm);

    mlir::LogicalResult applyTiling(mlir::RewriterBase& builder, Logger log);

    mlir::FailureOr<SmallVector<mlir::Operation*>> applyVerticalFusion(mlir::RewriterBase& builder, Logger log);

private:
    std::unique_ptr<ITilingAlgorithm> _tilingAlgorithm;
    mlir::Operation* _operation = nullptr;
};

// create and configure tiling context
TilingContext createTilingContext(mlir::Operation* operation, bool enableSCFTiling);
}  // namespace VPU
}  // namespace vpux
