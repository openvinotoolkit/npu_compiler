//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/pipelines/backend_pipeline_strategy.hpp"

namespace vpux {

//
// BackendPipelineStrategy50XX
//

class BackendPipelineStrategy50XX final : public IBackendPipelineStrategy {
public:
    void buildELFPipeline(mlir::OpPassManager& pm, const vpux::OV::Config& config, mlir::TimingScope& rootTiming,
                          Logger log) final;
};

}  // namespace vpux
