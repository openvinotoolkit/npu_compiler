//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/logger/logger.hpp"

#include <mlir/Pass/Pass.h>

#include <memory>
#include <optional>

namespace vpux {
namespace ELF {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAddELFSymbolTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddELFRelocationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddRelocationsForDynamicStridesDMAsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetOpOffsetsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetEntryPointPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddNetworkMetadataPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpdateELFSectionFlagsPass(Logger log = Logger::global(),
                                                            bool enableShaveDDRAccess = true);
std::unique_ptr<mlir::Pass> createRemoveEmptyELFSectionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleAlignmentRequirementsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddABIVersionPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetCMXSymbolValuePass(Logger log = Logger::global(),
                                                        std::optional<uint32_t> workspaceAddr = std::nullopt,
                                                        std::optional<uint32_t> workspaceSize = std::nullopt,
                                                        std::optional<uint32_t> metadataAddr = std::nullopt,
                                                        std::optional<uint32_t> metadataSize = std::nullopt);
std::unique_ptr<mlir::Pass> createFinalizeSkipDmaChainsPass(Logger log = Logger::global());

//
// Registration
//

void registerPasses();

}  // namespace ELF
}  // namespace vpux
