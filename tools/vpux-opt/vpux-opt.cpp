//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/init/dialects_registry.hpp"
#include "vpux/compiler/init/tool_registration.hpp"
#include "vpux/compiler/tools/options.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <cstdlib>

int main(int argc, char* argv[]) {
    vpux::Logger::setBaseStream(llvm::errs());
    try {
        // TODO: need to rework this unconditional replacement for dummy ops
        // there is an option for vpux-translate we can do it in the same way
        // Ticket: E#50937
        auto registry = vpux::createDialectRegistry(vpux::DummyOpMode::ENABLED);
        vpux::registerAllPassesGlobally();
        if (auto platform = vpux::parsePlatform(argc, argv); platform.has_value()) {
            vpux::registerAllHwSpecificComponents(registry, platform.value());
        }
        return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "NPU Optimizer Testing Tool", registry));
    } catch (const std::exception& e) {
        llvm::errs() << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
