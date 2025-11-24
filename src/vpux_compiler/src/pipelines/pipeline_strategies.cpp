//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/force_link_macros.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

// TODO: E#162744 remove this
DECLARE_FORCE_LINK(WsUtils);

namespace {
[[maybe_unused]] static const auto forceLinkRefs = [] {
    // use a symbol from another library here to ensure linker has to find it for this library (otherwise, the symbols
    // from WsUtils would leak into higher level libs)
    FORCE_LINK(WsUtils);
    return 0;
}();
}  // namespace
namespace vpux {

//
// DefaultHwStrategy
//

void DefaultHwStrategy::buildPipeline(mlir::OpPassManager& pm) {
    auto strategy = _createPipelineStrategy(config::CompilationMode::DefaultHW);

    strategy->initializePipeline(pm, _log);
    strategy->buildIEPipeline(pm, _log);
    strategy->buildLowerIE2VPUPipeline(pm, _log);
    strategy->buildVPUPipeline(pm, _log);
    strategy->buildLowerVPU2VPUIPPipeline(pm, _log);
    strategy->buildVPUIPPipeline(pm, _log);
}

//
// ReferenceSWStrategy
//

void ReferenceSWStrategy::buildPipeline(mlir::OpPassManager& pm) {
    auto strategy = _createPipelineStrategy(config::CompilationMode::ReferenceSW);

    strategy->initializePipeline(pm, _log);
    strategy->buildIEPipeline(pm, _log);
    strategy->buildLowerIE2VPUPipeline(pm, _log);
    strategy->buildVPUPipeline(pm, _log);
    strategy->buildLowerVPU2VPUIPPipeline(pm, _log);
    strategy->buildVPUIPPipeline(pm, _log);
}

//
// HostPipelineStrategy
//

void HostPipelineStrategy::buildPipeline(mlir::OpPassManager& pm) {
    auto strategy = _createPipelineStrategy(config::CompilationMode::HostCompile);

    strategy->initializePipeline(pm, _log);

    strategy->buildIEPipeline(pm, _log);
    strategy->buildLowerIE2VPUPipeline(pm, _log);
    strategy->buildVPUPipeline(pm, _log);

    pm.addPass(vpux::VPU::createFinalizeComputeFunctionBoundariesPass(_log));

    strategy->buildLowerVPU2VPUIPPipeline(pm, _log);

    pm.addPass(vpux::HostExec::createOptimizeMemRefCopiesPass(_log));
    pm.addPass(vpux::Core::createPackNestedModulesPass(_log));

    // dynamic shape optimizations
    pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
    pm.addPass(mlir::createCSEPass());

    // introduction of scratch buffer
    pm.addPass(vpux::HostExec::createReplaceAllocsWithSingleAllocAndViewsPass(_log));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    pm.addPass(vpux::HostExec::createPrepareHostFuncForAsyncExecutionPass(_log));

    auto& nestedPm = pm.nest<mlir::ModuleOp>();
    {
        nestedPm.addPass(vpux::Core::createAddNetInfoToModulePass(_log));
        strategy->initializePipeline(nestedPm, _log);

        strategy->buildVPUIPPipeline(nestedPm, _log);
    }
}

//
// createPipelineFactory
//

std::unique_ptr<IFrontendPipelineStrategy> createPipelineFactory(config::CompilationMode compilationMode,
                                                                 StrategyFactoryFn createPipelineStrategy, Logger log) {
    switch (compilationMode) {
    case config::CompilationMode::DefaultHW:
        return std::make_unique<DefaultHwStrategy>(createPipelineStrategy, log);
    case config::CompilationMode::ReferenceSW:
        return std::make_unique<ReferenceSWStrategy>(createPipelineStrategy, log);
    case config::CompilationMode::HostCompile:
        return std::make_unique<HostPipelineStrategy>(createPipelineStrategy, log);
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

}  // namespace vpux
