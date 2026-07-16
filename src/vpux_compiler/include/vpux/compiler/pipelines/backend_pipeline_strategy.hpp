//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/utils/logger/logger.hpp"
#include "vpux/utils/ov/config.hpp"

#include <mlir/Pass/PassManager.h>

namespace vpux {

//
// This strategy is responsible for building a "backend" pipeline.
//

class IBackendPipelineStrategy {
public:
    virtual void buildELFPipeline(mlir::OpPassManager& pm, const intel_npu::Config& config,
                                  mlir::TimingScope& rootTiming, Logger log) = 0;

    virtual ~IBackendPipelineStrategy() = default;
};

}  // namespace vpux
