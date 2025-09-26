//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"

namespace vpux::VPUIP::arch40xx {

class UnrollPermuteDMAStrategy : public IGreedilyPassStrategy {
public:
    UnrollPermuteDMAStrategy(int64_t dmaPortCount);

    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const final;

private:
    int64_t _dmaPortCount;
};

}  // namespace vpux::VPUIP::arch40xx
