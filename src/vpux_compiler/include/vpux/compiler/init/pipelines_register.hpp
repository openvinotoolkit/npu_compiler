//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

namespace vpux::config {
enum class Platform : uint64_t;
}

namespace vpux {

//
// IPipelineRegistry
//

class IPipelineRegistry {
public:
    virtual void registerPipelines() = 0;

    virtual ~IPipelineRegistry() = default;
};

//
// createPipelineRegistry
//

std::unique_ptr<IPipelineRegistry> createPipelineRegistry(config::Platform platform);

}  // namespace vpux
