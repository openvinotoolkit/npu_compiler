//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/dynamic_rewriter_strategies.hpp"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace vpux::IE {

//
// AdjustForVPUPipelineStrategy
//

class AdjustForVPUPipelineStrategy final : public IDynamicRewriterStrategy {
public:
    explicit AdjustForVPUPipelineStrategy(bool enableFuseClamp): _enableFuseClamp(enableFuseClamp) {
    }

    void registerRewriters(RewriterRegistry& registry, Logger& log) const override;

private:
    bool _enableFuseClamp = false;
};

std::unique_ptr<IDynamicRewriterStrategy> createAdjustForVPUPipelineStrategy(mlir::func::FuncOp funcOp,
                                                                             bool enableFuseClamp = false);

}  // namespace vpux::IE
