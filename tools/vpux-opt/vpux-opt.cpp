//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/NPUReg40XX/passes.hpp"
#include "vpux/compiler/NPU50XX/dialect/NPUReg50XX/passes.hpp"
#include "vpux/compiler/ShaveCodeGen/passes.hpp"
#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/transforms/passes.hpp"
#include "vpux/compiler/dialect/ELFNPU37XX/passes.hpp"
#include "vpux/compiler/dialect/HostExec/transforms/passes.hpp"
#include "vpux/compiler/dialect/IE/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUASM/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIPDPU/passes.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/passes.hpp"
#include "vpux/compiler/dialect/VPUMI40XX/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURegMapped/passes.hpp"
#include "vpux/compiler/dialect/config/constraints_initializer.hpp"
#include "vpux/compiler/dialect/const/passes.hpp"
#include "vpux/compiler/dialect/core/transforms/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/passes.hpp"
#include "vpux/compiler/dynamic_rewriter/rewriters_register.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/passes_register.hpp"
#include "vpux/compiler/pipelines_register.hpp"
#include "vpux/compiler/tools/options.hpp"

#include "vpux/utils/core/error.hpp"

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
    try {
        // TODO: need to rework this unconditional replacement for dummy ops
        // there is an option for vpux-translate we can do it in the same way
        // Ticket: E#50937
        auto registry = vpux::createDialectRegistry(vpux::DummyOpMode::ENABLED);

        const auto hwSpecificRegistration = [&](vpux::StringRef helpHeader) {
            const auto archKind = vpux::parseArchKind(argc, argv, helpHeader);

            const auto pipelineRegistry = vpux::createPipelineRegistry(archKind);
            pipelineRegistry->registerPipelines();

            const auto passesRegistry = vpux::createPassesRegistry(archKind);
            passesRegistry->registerPasses();

            auto interfacesRegistry = vpux::createInterfacesRegistry(archKind);
            interfacesRegistry->registerInterfaces(registry);

            vpux::config::registerConstraints(registry, archKind);
        };
        // Rewriter Registration
        vpux::createRewriterRegistry();
        // Pass Registration
        vpux::Core::registerPasses();
        vpux::Const::registerPasses();
        vpux::IE::registerPasses();
        vpux::IE::registerIEPipelines();
        vpux::VPU::registerPasses();
        vpux::VPU::registerVPUPipelines();
        vpux::VPUIP::registerPasses();
        vpux::VPUIP::registerVPUIPPipelines();
        vpux::HostExec::registerPasses();
        vpux::HostExec::registerHostExecPipelines();
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
        mlir::memref::registerResolveShapedTypeResultDims();
        mlir::registerLinalgPasses();
        mlir::memref::registerExpandStridedMetadataPass();
        mlir::registerArithToLLVMConversionPass();
        mlir::registerSCFToControlFlow();
        mlir::registerConvertControlFlowToLLVMPass();

        return mlir::asMainReturnCode(
                mlir::MlirOptMain(argc, argv, "NPU Optimizer Testing Tool", registry, hwSpecificRegistration));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
