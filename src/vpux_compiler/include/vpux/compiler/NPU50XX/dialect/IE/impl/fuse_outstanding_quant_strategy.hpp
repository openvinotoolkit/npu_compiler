//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::IE::arch50xx {

class FuseOutstandingQuantStrategy : public IGreedilyPassStrategy {
public:
    FuseOutstandingQuantStrategy() = default;

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;
};

}  // namespace vpux::IE::arch50xx
