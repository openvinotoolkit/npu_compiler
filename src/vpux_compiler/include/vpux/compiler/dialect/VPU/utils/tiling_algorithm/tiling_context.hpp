//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/tiling_algorithm/tiling_alg_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vf_merge_configuration.hpp"

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

    SmallVector<mlir::Operation*> applySCFTilingAndFusion(mlir::RewriterBase& builder,
                                                          const MergeConfiguration& mergeConfig, Logger log);

private:
    std::unique_ptr<ITilingAlgorithm> _tilingAlgorithm;
    mlir::Operation* _operation = nullptr;
};

struct TilingContextOptions {
    enum class ContextType { TILING, MULTICLUSTERING };
    ContextType type = ContextType::TILING;
    bool enableSCFTiling = false;

    TilingContextOptions(const ContextType& type, bool enableSCFTiling)
            : type(type), enableSCFTiling(enableSCFTiling) {};
};

// create and configure tiling context
TilingContext createTilingContext(mlir::Operation* operation, const TilingContextOptions& options);
}  // namespace VPU
}  // namespace vpux
