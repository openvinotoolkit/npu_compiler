//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/pipelines_register.hpp"

namespace vpux {

//
// PipelineRegistry50XX
//

class PipelineRegistry50XX final : public IPipelineRegistry {
public:
    void registerPipelines() override;
};

}  // namespace vpux
