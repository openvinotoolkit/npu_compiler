//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/utils/gather_dma_utils.hpp"

#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux::VPU::arch40xx {

/*
   Class for getting ConvertOpToDMAForPerformantExecutionStrategy patterns for VPU40XX
*/
class ConvertOpToDMAForPerformantExecutionStrategy : public IConversionPassStrategy {
public:
    void addPatterns(mlir::RewritePatternSet& patterns, Logger& log) const override;
    void markOpLegality(mlir::ConversionTarget& target, Logger& log) const override;
};

}  // namespace vpux::VPU::arch40xx
