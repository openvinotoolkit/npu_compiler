//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::net {

namespace {
void addBlockToEmptyRegion(mlir::Region& region) {
    if (region.empty()) {
        region.emplaceBlock();
    }
}
}  // namespace

void setupSections(net::NetworkInfoOp netInfo, bool enableProfiling) {
    addBlockToEmptyRegion(netInfo.getInputsInfo());
    addBlockToEmptyRegion(netInfo.getOutputsInfo());
    if (enableProfiling) {
        VPUX_THROW_WHEN(netInfo.getProfilingOutputsInfo().empty(), "profiling_outputs_info is not created");
        addBlockToEmptyRegion(netInfo.getProfilingOutputsInfo().front());
    }
}

void eraseSectionEntries(mlir::Region& section, size_t begin) {
    VPUX_THROW_WHEN(section.getBlocks().size() != 1, "A section must have exactly 1 block");

    auto& block = section.front();
    VPUX_THROW_WHEN(block.getOperations().size() < begin, "Too many entries asked to be erased");
    for (size_t n = block.getOperations().size() - begin; n > 0; --n) {
        block.back().erase();
    }
}

mlir::func::FuncOp findEntryPointFunc(mlir::Operation* op, Logger& log) {
    if (op == nullptr) {
        return nullptr;
    }

    auto topModuleOp = getTopParentOpOfType<mlir::ModuleOp>(op);
    if (topModuleOp == nullptr) {
        log.warning("Top level module not found");
        return nullptr;
    }

    auto netOps = to_small_vector(topModuleOp.getOps<net::NetworkInfoOp>());
    if (netOps.size() != 1) {
        log.warning("Expected exactly one net::NetworkInfoOp, found {0}", netOps.size());
        return nullptr;
    }

    auto netInfo = netOps.front();
    auto netFunc = topModuleOp.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPointAttr());
    if (netFunc == nullptr) {
        log.warning("Entry point function '{0}' not found", netInfo.getEntryPoint());
        return nullptr;
    }

    return netFunc;
}

}  // namespace vpux::net
