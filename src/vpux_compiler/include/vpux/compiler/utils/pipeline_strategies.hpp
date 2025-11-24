//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/pipelines/frontend_pipeline_strategy.hpp"

namespace vpux {

//
// HW-agnostic high-level pipelines:
//

class DefaultHwStrategy final : public IFrontendPipelineStrategy {
public:
    using IFrontendPipelineStrategy::IFrontendPipelineStrategy;

    void buildPipeline(mlir::OpPassManager& pm) final;
};

class ReferenceSWStrategy final : public IFrontendPipelineStrategy {
public:
    using IFrontendPipelineStrategy::IFrontendPipelineStrategy;

    void buildPipeline(mlir::OpPassManager& pm) final;
};

class HostPipelineStrategy final : public IFrontendPipelineStrategy {
public:
    using IFrontendPipelineStrategy::IFrontendPipelineStrategy;

    void buildPipeline(mlir::OpPassManager& pm) final;
};

//
// createPipelineFactory
//

std::unique_ptr<IFrontendPipelineStrategy> createPipelineFactory(config::CompilationMode compilationMode,
                                                                 StrategyFactoryFn createPipelineStrategy, Logger log);

}  // namespace vpux
