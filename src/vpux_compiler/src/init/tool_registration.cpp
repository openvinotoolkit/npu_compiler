//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/tool_registration.hpp"

// NPU passes
#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp"
#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/rewriters.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/bytecode/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/passes.hpp"

// NPU HW-specific components
#include "vpux/compiler/init/hw_strategy_registry.hpp"
#include "vpux/compiler/init/interfaces_registry.hpp"
#include "vpux/compiler/init/passes_register.hpp"
#include "vpux/compiler/init/pipelines_register.hpp"
#include "vpux/compiler/init/singleton_initializer.hpp"
#include "vpux/compiler/locverif/passes.hpp"

// MLIR passes and components
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Quant/Transforms/Passes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

namespace vpux {
void registerAllPassesGlobally() {
    // Rewriter Registration
    auto& registry = RegistryManager::getGlobalRegistry();
    vpux::VPUIP::registerVPUIPRewriters(registry);
    // Pass Registration
    vpux::Core::registerPasses();
    vpux::locverif::registerPasses();
    vpux::Const::registerPasses();
    vpux::IE::registerPasses();
    vpux::IE::registerIEPipelines();
    vpux::VPU::registerPasses();
    vpux::VPU::registerVPUPipelines();
    vpux::VPUIP::registerPasses();
    vpux::VPUIP::registerVPUIPPipelines();
    vpux::HostExec::registerPasses();
    vpux::HostExec::registerHostExecPipelines();
    vpux::bytecode::registerPasses();
    vpux::VPURT::registerVPURTPipelines();
    vpux::VPURT::registerPasses();
    vpux::ELFNPU37XX::registerPasses();
    vpux::ELF::registerPasses();
    vpux::VPUMI37XX::registerPasses();
    vpux::VPUMI40XX::registerPasses();
    vpux::NPUReg40XX::registerPasses();
    vpux::VPUASM::registerPasses();
    vpux::VPUIPDPU::registerPasses();
    vpux::ShaveCodeGen::registerPasses();
    vpux::registerConversionPasses();
    vpux::registerConversionPipelines();
    vpux::registerDynamicRewriterExecutorPass();

    vpux::NPUReg50XX::registerPasses();

    mlir::registerTransformsPasses();
    mlir::func::registerFuncPasses();
    mlir::memref::registerResolveShapedTypeResultDimsPass();
    mlir::registerLinalgPasses();
    mlir::memref::registerExpandStridedMetadataPass();
    mlir::registerArithToLLVMConversionPass();
    mlir::registerSCFToControlFlowPass();
    mlir::registerConvertControlFlowToLLVMPass();
    mlir::quant::registerQuantPasses();
}

void registerAllHwSpecificComponents(mlir::DialectRegistry& registry, vpux::config::Platform platform) {
    const auto pipelineRegistry = vpux::createPipelineRegistry(platform);
    pipelineRegistry->registerPipelines();

    const auto passesRegistry = vpux::createPassesRegistry(platform);
    passesRegistry->registerPasses();

    auto interfacesRegistry = vpux::createInterfacesRegistry(platform);
    interfacesRegistry->registerInterfaces(registry);

    vpux::config::registerConstraints(registry, platform);
    vpux::IE::registerStrategies(registry, platform);
    vpux::VPU::initializeSingletons(registry, platform);
    vpux::VPU::registerStrategies(registry, platform);
    vpux::VPUIP::registerStrategies(registry, platform);
}
}  // namespace vpux
