//
// Copyright (C) 2022-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/core/profiling.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"

#include <mlir/IR/Visitors.h>

using namespace vpux;

namespace {

template <class T, typename = require_t<std::is_integral<T>>>
mlir::IntegerAttr getOptionalInt(mlir::MLIRContext* ctx, std::optional<T> value) {
    return value.has_value() ? getIntAttr(ctx, value.value()) : nullptr;
}

};  // namespace

VPUIP::DpuProfilingMetadataAttr vpux::getDpuProfilingMetaAttr(mlir::MLIRContext* ctx, unsigned bufferId,
                                                              unsigned taskId, unsigned maxVariants,
                                                              unsigned numVariants, unsigned clusterId) {
    return VPUIP::DpuProfilingMetadataAttr::get(ctx, getIntAttr(ctx, bufferId), getIntAttr(ctx, taskId),
                                                getIntAttr(ctx, maxVariants), getIntAttr(ctx, numVariants),
                                                getIntAttr(ctx, clusterId));
}

VPUIP::DmaProfilingMetadataAttr vpux::getDmaProfilingMetaAttrBegin(mlir::MLIRContext* ctx) {
    return VPUIP::DmaProfilingMetadataAttr::get(ctx, /*dataIndex=*/nullptr, mlir::UnitAttr::get(ctx));
}

VPUIP::DmaProfilingMetadataAttr vpux::getDmaProfilingMetaAttr(mlir::MLIRContext* ctx, unsigned index) {
    return VPUIP::DmaProfilingMetadataAttr::get(ctx, getIntAttr(ctx, index), /*profBegin=*/nullptr);
}

VPUIP::SwProfilingMetadataAttr vpux::getSwProfilingMetaAttr(mlir::MLIRContext* ctx, size_t bufferId,
                                                            size_t bufferOffset, size_t clusterSize, size_t dataIndex,
                                                            size_t tileId, size_t clusterId) {
    return VPUIP::SwProfilingMetadataAttr::get(ctx, getIntAttr(ctx, bufferId), getIntAttr(ctx, bufferOffset),
                                               getIntAttr(ctx, clusterSize), getIntAttr(ctx, dataIndex),
                                               getIntAttr(ctx, tileId), getIntAttr(ctx, clusterId));
}

DMAProfilingMode vpux::getDMAProfilingMode(config::ArchKind arch, const std::string& optionValue) {
    if (optionValue == "false") {
        return arch == config::ArchKind::NPU40XX ? DMAProfilingMode::SCRATCH : DMAProfilingMode::DISABLED;
    }
    switch (arch) {
    case config::ArchKind::NPU37XX:
        if (optionValue == "true") {
            return DMAProfilingMode::SW;
        } else {
            VPUX_THROW("Unsupported dma-profiling option value: {0}", optionValue);
        }
    default:
        if (optionValue == "true") {
            return DMAProfilingMode::DYNAMIC_HWP;
        } else if (optionValue == "static") {
            return DMAProfilingMode::STATIC_HWP;
        } else {
            VPUX_THROW("Unsupported dma-profiling option value: {0}", optionValue);
        }
    }
}

mlir::BlockArgument vpux::addNewProfilingOutput(mlir::MLIRContext* ctx, mlir::func::FuncOp& netFunc,
                                                net::NetworkInfoOp& netInfo, mlir::MemRefType outputType,
                                                profiling::ExecutorType execType) {
    const auto name = convertExecTypeToName(execType);
    //
    // Declare and create additional output from network
    //
    auto funcType = netFunc.getFunctionType();
    auto newResultTypes = to_small_vector(llvm::concat<const mlir::Type>(funcType.getResults(), ArrayRef(outputType)));
    auto newInputsTypes = to_small_vector(llvm::concat<const mlir::Type>(funcType.getInputs(), ArrayRef(outputType)));

    auto newFunctionType = mlir::FunctionType::get(ctx, newInputsTypes, newResultTypes);
    netFunc.setType(newFunctionType);

    // If you hit this, IR have NetworkInfoOp without profilingOutputInfo region
    VPUX_THROW_WHEN(netInfo.getProfilingOutputsInfo().empty(), "Could not add profiling output: no region added");

    const auto ndOutputType = mlir::cast<vpux::NDTypeInterface>(outputType);

    // Adding output to the user info
    auto outputUserResult =
            getTensorType(ndOutputType.getShape(), ndOutputType.getElementType(), ndOutputType.getDimsOrder(), nullptr);
    auto userInfoBuilder = mlir::OpBuilder::atBlockEnd(&netInfo.getProfilingOutputsInfo().front().front());
    net::DataInfoOp::create(userInfoBuilder, mlir::NameLoc::get(mlir::StringAttr::get(ctx, "profilingDataOutputInfo")),
                            name, outputUserResult);

    const mlir::Location suffixLoc = mlir::NameLoc::get(mlir::StringAttr::get(ctx, "profiling_" + name));
    const auto argLoc = mlir::FusedLoc::get(ctx, {netFunc.getLoc(), suffixLoc});

    return netFunc.getBody().front().addArgument(outputType, argLoc);
}

bool vpux::isProfiledDmaTask(VPURT::TaskOp taskOp) {
    auto* wrappedTaskOp = taskOp.getInnerTaskOp();

    return mlir::isa_and_nonnull<VPUIP::DMATypeOpInterface>(wrappedTaskOp) &&
           !mlir::isa<VPUIP::SyncDMAOp>(wrappedTaskOp);
}

void vpux::setDmaHwpIdAttribute(mlir::MLIRContext* ctx, VPUIP::DMATypeOpInterface& op, int32_t dmaHwpId) {
    auto dmaHwpIdAttrib = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32, mlir::IntegerType::Signed), dmaHwpId);
    op.setDmaHwpIdAttr(dmaHwpIdAttrib);
}

bool vpux::isDmaHwpUsedInVPURT(mlir::func::FuncOp& func) {
    bool dmaHwpEnabled = false;
    func->walk([&](VPURT::TaskOp taskOp) {
        if (!vpux::isProfiledDmaTask(taskOp)) {
            return mlir::WalkResult::interrupt();
        }

        auto op = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(taskOp.getInnerTaskOp());
        if (op && op.getDmaHwpIdAttr() != nullptr) {
            dmaHwpEnabled = true;
            return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
    });
    return dmaHwpEnabled;
}

bool vpux::isDmaHwpUsedInVPURT(mlir::ModuleOp& module) {
    if (vpux::config::getArch(module) < vpux::config::ArchKind::NPU40XX) {
        return false;
    }
    auto func = net::getMainFunc(module);
    return vpux::isDmaHwpUsedInVPURT(func);
}

bool vpux::isProfilingEnabled(net::NetworkInfoOp netInfo) {
    auto profilingOutputsInfo = netInfo.getProfilingOutputsDataInfo();
    VPUX_THROW_WHEN(profilingOutputsInfo.size() > 1, "Unexpected number of profiling outputs (expected 1, got {0})",
                    profilingOutputsInfo.size());
    return profilingOutputsInfo.size() > 0;
}

std::optional<VPUIP::ProfilingSectionOp> vpux::getProfilingSection(mlir::ModuleOp module,
                                                                   profiling::ExecutorType secType) {
    if (module.getOps<net::NetworkInfoOp>().empty()) {
        return {};
    }
    auto [netInfo, netFunc] = net::getFromModule(module);
    if (!isProfilingEnabled(netInfo)) {
        return {};
    }
    auto profilingOutputInfo = *netInfo.getProfilingOutputsInfo().front().getOps<net::DataInfoOp>().begin();

    auto& sections = profilingOutputInfo.getSections().front().front();
    for (VPUIP::ProfilingSectionOp section : sections.getOps<VPUIP::ProfilingSectionOp>()) {
        if (static_cast<profiling::ExecutorType>(section.getSectionType()) == secType) {
            return section;
        }
    }
    return {};
}
