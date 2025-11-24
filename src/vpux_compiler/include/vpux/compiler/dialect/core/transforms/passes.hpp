//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/logger/logger.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

namespace vpux::Core {

//
// Passes
//

std::unique_ptr<mlir::Pass> createMoveDeclarationsToTopPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createPrintDotPass(StringRef fileName = {}, StringRef startAfter = {},
                                               StringRef stopBefore = {}, bool printOnlyDotInterFaces = false,
                                               bool printConst = false, bool printDeclarations = false);

std::unique_ptr<mlir::Pass> createSetupLocationVerifierPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createStartLocationVerifierPass(
        vpux::Logger log, const mlir::detail::PassOptions::Option<std::string>& locationsVerificationMode);
std::unique_ptr<mlir::Pass> createStopLocationVerifierPass(vpux::Logger log);
std::unique_ptr<mlir::Pass> createPackNestedModulesPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUnpackNestedModulesPass(const Logger& log = Logger::global());
std::unique_ptr<mlir::Pass> createAddNetInfoToModulePass(Logger log = Logger::global(),
                                                         bool hasTensorSemantics = false);

//
// Registration
//

void registerPasses();

}  // namespace vpux::Core
