//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::IE::arch37xx {

class FuseOutstandingDequantStrategy : public IGreedilyPassStrategy {
public:
    FuseOutstandingDequantStrategy() = default;

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;
};

}  // namespace vpux::IE::arch37xx
