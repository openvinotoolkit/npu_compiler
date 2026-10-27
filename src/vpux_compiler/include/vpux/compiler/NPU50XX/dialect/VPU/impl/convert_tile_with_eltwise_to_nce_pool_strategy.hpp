//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::VPU::arch50xx {

class ConvertTileWithEltwiseToNCEPoolStrategy final : public IIterativeWalkPassStrategy {
public:
    explicit ConvertTileWithEltwiseToNCEPoolStrategy(mlir::MLIRContext* ctx): _ctx(ctx) {
    }

    void addPatterns(SmallVector<mlir::RewritePatternSet>& patterns, Logger& log) const final;

private:
    mlir::MLIRContext* _ctx;
};

}  // namespace vpux::VPU::arch50xx
