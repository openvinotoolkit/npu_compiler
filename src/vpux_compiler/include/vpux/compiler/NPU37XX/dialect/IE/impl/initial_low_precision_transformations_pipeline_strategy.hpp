//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

namespace vpux::IE::arch37xx {

//
// InitialLowPrecisionTransformationsPipelineStrategy
//

class InitialLowPrecisionTransformationsPipelineStrategy final : public IDynamicRewriterStrategy {
public:
    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;
};

}  // namespace vpux::IE::arch37xx
