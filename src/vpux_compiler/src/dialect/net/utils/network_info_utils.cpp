//
// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/dialect/core/IR/strided_dmas_utils.hpp"
#include "vpux/compiler/dialect/core/types.hpp"
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

net::NetworkInfoOp getNetworkInfo(mlir::Operation* op) {
    auto moduleOp = vpux::getModuleOp(op);

    auto netOps = moduleOp.getOps<net::NetworkInfoOp>();
    assert(std::distance(netOps.begin(), netOps.end()) == 1 && "Must have exactly 1 'net::NetworkInfoOp' in Module");
    return *netOps.begin();
}

mlir::func::FuncOp getMainFunc(mlir::Operation* op) {
    auto moduleOp = getModuleOp(op);
    auto netInfo = getNetworkInfo(moduleOp);
    auto netFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPointAttr());

    assert(netFunc != nullptr && "Can't find entryPoint");

    return netFunc;
}

std::pair<net::NetworkInfoOp, mlir::func::FuncOp> getFromModule(mlir::Operation* op) {
    auto moduleOp = getModuleOp(op);
    auto netInfo = getNetworkInfo(moduleOp);
    auto netFunc = moduleOp.lookupSymbol<mlir::func::FuncOp>(netInfo.getEntryPointAttr());

    assert(netFunc != nullptr && "Can't find entryPoint");

    return {netInfo, netFunc};
}

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

SmallVector<int64_t> getBoundsFromDataInfo(ArrayRef<net::DataInfoOp> dataInfos, size_t index) {
    if (index >= dataInfos.size()) {
        return {};
    }

    auto dataInfo = dataInfos[index];
    const auto userType = dataInfo.getUserType();
    const auto boundedType = mlir::dyn_cast<Core::BoundedTensorType>(userType);
    if (boundedType == nullptr) {
        return {};
    }

    return to_small_vector(boundedType.getBounds());
}

bool isArgStrided(mlir::ModuleOp module, size_t argIndex) {
    auto netInfos = module.getOps<net::NetworkInfoOp>();
    if (netInfos.empty()) {
        return false;
    }

    auto netInfo = *(netInfos.begin());
    auto inputs = to_small_vector(netInfo.getInputsInfo().getOps<net::DataInfoOp>());
    if (argIndex < inputs.size()) {
        return inputs[argIndex]->getAttr(vpux::dynamicStridesAttrName) != nullptr;
    } else {
        auto outputs = to_small_vector(netInfo.getOutputsInfo().getOps<net::DataInfoOp>());
        auto outIdx = argIndex - inputs.size();
        if (outIdx < outputs.size()) {
            return outputs[outIdx]->getAttr(vpux::dynamicStridesAttrName) != nullptr;
        }
    }

    return false;
}

}  // namespace vpux::net
