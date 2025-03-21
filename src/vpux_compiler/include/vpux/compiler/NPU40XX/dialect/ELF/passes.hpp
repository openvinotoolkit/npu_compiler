//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/NPU40XX/dialect/ELF/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/dialect.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/passes.hpp"

#include "vpux/utils/core/logger.hpp"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace vpux {
namespace ELF {

//
// Passes
//

std::unique_ptr<mlir::Pass> createAddELFSymbolTablePass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddELFRelocationsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createSetOpOffsetsPass(Logger log = Logger::global(),
                                                   bool enableComputeTaskBufferOffsets = false);
std::unique_ptr<mlir::Pass> createSetEntryPointPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddNetworkMetadataPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createUpdateELFSectionFlagsPass(Logger log = Logger::global(),
                                                            std::string isShaveDDRAccessEnabled = "true");
std::unique_ptr<mlir::Pass> createRemoveEmptyELFSectionsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createHandleAlignmentRequirementsPass(Logger log = Logger::global());
std::unique_ptr<mlir::Pass> createAddABIVersionPass(Logger log = Logger::global(), uint32_t versionMajor = 0,
                                                    uint32_t versionMinor = 0, uint32_t versionPatch = 0);

//
// Registration
//

void registerPasses();

}  // namespace ELF
}  // namespace vpux
