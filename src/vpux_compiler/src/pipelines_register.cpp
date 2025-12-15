//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/pipelines_register.hpp"
#include "vpux/compiler/NPU40XX/pipelines_register.hpp"
#include "vpux/compiler/NPU50XX/pipelines_register.hpp"

#include "vpux/utils/core/error.hpp"

using namespace vpux;

//
// createPipelineRegistry
//

std::unique_ptr<IPipelineRegistry> vpux::createPipelineRegistry(config::ArchKind arch) {
    switch (arch) {
    case config::ArchKind::NPU37XX:
        return std::make_unique<PipelineRegistry37XX>();
    case config::ArchKind::NPU40XX:
        return std::make_unique<PipelineRegistry40XX>();
    case config::ArchKind::NPU50XX:
        return std::make_unique<PipelineRegistry50XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}
