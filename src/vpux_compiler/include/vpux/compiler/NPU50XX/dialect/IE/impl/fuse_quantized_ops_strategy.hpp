//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::IE::arch50xx {

class FuseQuantizedOpsStrategy : public IGreedilyPassStrategy {
public:
    FuseQuantizedOpsStrategy(const bool seOpsEnabled, const bool seExperimentalOpsEnabled)
            : _seOpsEnabled(seOpsEnabled), _seExperimentalOpsEnabled(seExperimentalOpsEnabled) {
    }

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override final;

private:
    bool _seOpsEnabled;
    bool _seExperimentalOpsEnabled;
};

}  // namespace vpux::IE::arch50xx
