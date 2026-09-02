//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*
 * This test aims to fuzz the DefaultHWMode pipeline.
 * It receives an IE dialect IR that is passed through the InitCompiler pass, followed by the DefaultHWMode pipeline.
 */

#include "vpux/compiler/NPU37XX/dialect_pipeline_strategy.hpp"
#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/attributes.hpp"
#include "vpux/compiler/init/dialects_registry.hpp"

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

using namespace vpux;

static mlir::MLIRContext ctx;
static mlir::DialectRegistry registry;

extern "C" int LLVMFuzzerInitialize(int* /*argc*/, char*** /*argv*/) {
    registry = createDialectRegistry();
    ctx.appendDialectRegistry(registry);

    // Disable multithreading to ensure all exceptions are caught
    ctx.disableMultithreading();
    // Ignore error messages encountered during parsing invalid IRs
    ctx.getDiagEngine().registerHandler([](mlir::Diagnostic&) -> mlir::LogicalResult {
        return mlir::success();
    });
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Data missing null terminator will assert in parser. Copying the data into a string will ensure it is
    // null-terminated and avoid the parsing issue.
    // This is done since the MLIR parser is not the scope of this test
    const std::string dataStr(reinterpret_cast<const char*>(data), size);

    // Also skip empty data for the same reason
    if (dataStr.empty()) {
        return -1;
    }

    try {
        const llvm::StringRef irStr(dataStr.c_str(), size);
        const auto moduleOp = mlir::parseSourceString<mlir::ModuleOp>(irStr, &ctx);
        if (!moduleOp) {
            // Unparsable IRs should not be added to the corpus, even if they increase the code coverage of the parser
            return -1;
        }

        const auto log = vpux::Logger::global();

        mlir::PassManager pm(moduleOp.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
        auto initCompilerOptions =
                VPU::InitCompilerOptions(config::ArchKind::NPU40XX, config::CompilationMode::DefaultHW);

        VPU::buildInitCompilerPipeline(pm, initCompilerOptions, log);

        const DefaultHWOptions37XX options;
        buildDefaultHWModePipeline(pm, options, log.nest());

        auto result = pm.run(*moduleOp);
        if (mlir::failed(result)) {
            return -1;
        }
    } catch (const ov::Exception&) {
        // Ignore OpenVINO exceptions (e.g. from OpenVINO shape infer functions used in return type infer of IE ops)
        return -1;
    } catch (const std::exception&) {
    }

    return 0;
}
