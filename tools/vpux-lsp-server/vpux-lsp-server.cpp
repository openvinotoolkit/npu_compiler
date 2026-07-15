//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/init/tool_registration.hpp"
#include "vpux/compiler/tools/options.hpp"

#include <mlir/Tools/mlir-lsp-server/MlirLspServerMain.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <cstdlib>

int main(int argc, char* argv[]) {
    try {
        auto registry = vpux::createDialectRegistry(vpux::DummyOpMode::ENABLED);
        vpux::registerAllPassesGlobally();
        if (auto platform = vpux::parsePlatform(argc, argv); platform.has_value()) {
            vpux::registerAllHwSpecificComponents(registry, platform.value());
        }

        return mlir::asMainReturnCode(mlir::MlirLspServerMain(argc, argv, registry));
    } catch (const std::exception& e) {
        llvm::errs() << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
