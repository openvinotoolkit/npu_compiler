//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/init/pipelines_register.hpp"

namespace vpux {

//
// PipelineRegistry50XX
//

class PipelineRegistry50XX final : public IPipelineRegistry {
public:
    explicit PipelineRegistry50XX(config::Platform platform);
    void registerPipelines() override;

private:
    config::Platform _platform;
};

}  // namespace vpux
