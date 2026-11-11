//
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/pipelines_register.hpp"

namespace vpux {

//
// PipelineRegistry60XX
//

class PipelineRegistry60XX final : public IPipelineRegistry {
public:
    explicit PipelineRegistry60XX(config::Platform platform);
    void registerPipelines() override;

private:
    config::Platform _platform;
};

}  // namespace vpux
