//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/backend_pipeline_strategy.hpp"

#include "vpux/compiler/NPU37XX/conversion.hpp"

using namespace vpux;

//
// BackendPipelineStrategy37XX::buildELFPipeline
//

void BackendPipelineStrategy37XX::buildELFPipeline(mlir::OpPassManager& pm, const vpux::OV::Config&,
                                                   mlir::TimingScope& rootTiming, Logger log) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");
    arch37xx::buildLowerVPUIP2ELFPipeline(pm, log.nest());
}
