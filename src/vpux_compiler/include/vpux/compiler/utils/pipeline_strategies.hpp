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

class WSMonolithicStrategy final : public IFrontendPipelineStrategy {
public:
    using IFrontendPipelineStrategy::IFrontendPipelineStrategy;

    void buildPipeline(mlir::OpPassManager& pm) final;

    /// @brief This is a special constructor that disables the VPUIP part of the WS Monolithic pipeline. This is used
    /// only for LIT tests to ease testing of the intermediate results before VPUIP.
    static WSMonolithicStrategy createForLITTests(StrategyFactoryFn createPipelineStrategy, Logger log) {
        auto result = WSMonolithicStrategy(std::move(createPipelineStrategy), log);
        result._disableVPUIP = true;
        return result;
    }

private:
    bool _disableVPUIP = false;
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
