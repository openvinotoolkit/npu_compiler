//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_factory.hpp"
#include "vpux/utils/logger/logger.hpp"

namespace vpux {
/**
 * Interface for implementing platform specific dynamic rewriter strategies
 */
class IDynamicRewriterStrategy {
public:
    virtual ~IDynamicRewriterStrategy() = default;

    virtual void registerRewriters(RewriterRegistry& registry, Logger& log) const = 0;
};

}  // namespace vpux
