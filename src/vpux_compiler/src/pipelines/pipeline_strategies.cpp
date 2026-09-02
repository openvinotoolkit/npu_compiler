//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"

#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/core/force_link_macros.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/pipeline_strategies.hpp"

#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
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
    strategy->buildDebatcherPipeline(pm, _log);
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
    strategy->buildDebatcherPipeline(pm, _log);
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

    // Fuse color conversion pattern BEFORE shape extraction so that
    // @output_shape sees the fused YuvToRgb (which has reifyResultShapes)
    // instead of the unfused chain of IE ops + const.Declare
    pm.addPass(IE::createFuseColorConversionPass(/*enableYuvToRgbShaveScale=*/true, _log));

    // Recover IE.Transpose from any ShapeOf/Gather/DynamicReshape decomposition
    // emitted by the frontend for fully-dynamic Transpose ops. IE.Transpose implements
    // ReifyRankedShapedTypeOpInterface, unlike the individual ops in that chain, which
    // is required by the output-shape prediction pipeline below.
    pm.addPass(IE::createFoldShapeOfGatherDynamicReshapeToTransposePass(_log));

    // build output shape predict func and pack @main func to a nested @NPU module
    HostExec::buildOutputShapePredictPipeline(pm, _log);

    strategy->buildDebatcherPipeline(pm, _log);

    // pack @NPU module
    // From now on, all host-compile related functions will be located in the top module
    pm.addPass(Core::createPackNestedModulesPass(_log, Core::NestingMode::EntryPoint));
    auto& nestedNPUPm = pm.nest<mlir::ModuleOp>();
    strategy->buildIEPipeline(nestedNPUPm, _log);
    strategy->buildLowerIE2VPUPipeline(nestedNPUPm, _log);
    strategy->buildVPUPipeline(nestedNPUPm, _log);
    // unpack @NPU module
    // Starting from that moment, all code processing each dynamic-dimension
    // must be extracted into pure host-compile functions, which will be put into
    // the top module
    pm.addPass(Core::createUnpackNestedModulesPass(_log, Core::NestingMode::EntryPoint));
    pm.addPass(VPU::createFinalizeComputeFunctionBoundariesPass(_log));
    // Clean up function boundaries from extra ops (e.g. ReinterpretCast)
    pm.addPass(mlir::createCanonicalizerPass());

    // Link all resolved function calls after module unpacking
    pm.addPass(vpux::HostExec::createWrapFuncCallPass(_log));

    strategy->buildLowerVPU2VPUIPPipeline(pm, _log);
    pm.addPass(vpux::HostExec::createPropagateDynamicShapesPass(_log));

    pm.addPass(vpux::HostExec::createOptimizeMemRefCopiesPass(_log));

    // dynamic shape optimizations
    pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
    pm.addPass(mlir::createCSEPass());

    // introduction of scratch buffer
    pm.addPass(vpux::HostExec::createReplaceAllocsWithSingleAllocAndViewsPass(_log));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(vpux::HostExec::createPrepareHostFuncForAsyncExecutionPass(/*removeReturnValues=*/true, _log));

    auto& nestedPm = pm.nest<mlir::ModuleOp>();
    { strategy->buildVPUIPPipeline(nestedPm, _log); }
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
    case config::CompilationMode::HostCompile_JIT:
    case config::CompilationMode::HostCompile_Interpreter:
        return std::make_unique<HostPipelineStrategy>(createPipelineStrategy, log);
    default:
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
}

}  // namespace vpux
