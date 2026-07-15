//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/pipelines_register.hpp"
#include "vpux/compiler/NPU40XX/pipelines_register.hpp"
#include "vpux/compiler/NPU50XX/pipelines_register.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// createPipelineRegistry
//

std::unique_ptr<IPipelineRegistry> vpux::createPipelineRegistry(config::Platform platform) {
    switch (platform) {
    case config::Platform::NPU3720:
        return std::make_unique<PipelineRegistry37XX>();
    case config::Platform::NPU4000:
        return std::make_unique<PipelineRegistry40XX>();
    case config::Platform::NPU5010:
    case config::Platform::NPU5020:
        return std::make_unique<PipelineRegistry50XX>(platform);
    default:
        VPUX_THROW("Unsupported platform: {0}", platform);
    }
}
