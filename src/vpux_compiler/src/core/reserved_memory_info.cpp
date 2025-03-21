//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/reserved_memory_info.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"

#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/DenseSet.h>

using namespace vpux;

ReservedMemInfo::ReservedMemInfo(mlir::ModuleOp moduleOp, mlir::AnalysisManager& am) {
    // TODO:#108991 -- for now only "main" function with inner functions is supported,
    // but it is possible support multiple nested calls using a loop through call/function ops
    mlir::func::FuncOp netFunc;
    IE::CNNNetworkOp netInfo;
    IE::CNNNetworkOp::getFromModule(moduleOp, netInfo, netFunc);

    auto liveRangeInfo = am.getChildAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::DDR>>(netFunc);
    auto scanResult = am.getChildAnalysis<AllocationInfo>(netFunc).getScanResult(VPU::MemoryKind::DDR);
    auto depsInfo = am.getChildAnalysis<AsyncDepsInfo>(netFunc);

    reserveFunctionRanges(netFunc, liveRangeInfo, scanResult);
    reserveModuleRanges(scanResult);
    linearizeCallOps(netFunc, depsInfo);
}

ReservedMemInfo::ReservedMemInfo(mlir::func::FuncOp netFunc) {
    AsyncDepsInfo depsInfo{netFunc};
    AliasesInfo aliasesInfo{netFunc};
    MemLiveRangeInfo liveRangeInfo{netFunc, aliasesInfo, VPU::MemoryKind::DDR};
    // AllocationInfo modifies MemLiveRangeInfo, provide a copy
    auto allocRangeInfo = liveRangeInfo;
    AllocationInfo allocInfo{netFunc, depsInfo, allocRangeInfo};
    auto scanResult = allocInfo.getScanResult(VPU::MemoryKind::DDR);

    reserveFunctionRanges(netFunc, liveRangeInfo, scanResult);
    reserveModuleRanges(scanResult);
    linearizeCallOps(netFunc, depsInfo);
}

ReservedMemInfo::ReservedAddressAndSizeVector ReservedMemInfo::getUniqueRanges(const ValueOrderedSet& buffers,
                                                                               ScanResult& scanResult) {
    auto& allReservedMemInfo = scanResult.linearScanHandler;
    ReservedAddressAndSizeVector allRanges;
    for (const auto& buffer : buffers) {
        if (mlir::isa<mlir::BlockArgument>(buffer)) {
            continue;
        }
        const auto address = allReservedMemInfo.getAddress(buffer);
        const auto size = allReservedMemInfo.getSize(buffer);
        allRanges.push_back(std::make_pair(address, size));
    }

    if (allRanges.size() < 2) {
        return allRanges;
    }

    // sort the operations to avoid O(n^2)
    llvm::sort(allRanges.begin(), allRanges.end(),
               [](const std::pair<vpux::AddressType, vpux::AddressType>& entry1,
                  const std::pair<vpux::AddressType, vpux::AddressType>& entry2) {
                   // first address begin
                   if (entry1.first != entry2.first) {
                       return entry1.first < entry2.first;
                   }

                   // second address end
                   if (entry1.second != entry2.second) {
                       return entry1.second < entry2.second;
                   }

                   // allow self comparison
                   return false;
               });

    ReservedAddressAndSizeVector uniqueRanges;

    auto isOverlapRange = [&](const AddressType address, const AddressType size) {
        // check if range exists
        for (auto& entry : uniqueRanges) {
            if (address + size < entry.first) {
                // range ends before
                continue;
            }
            if (entry.first + entry.second <= address) {
                // range starts after
                continue;
            }
            // overlap, extend range
            entry.first = std::min(entry.first, address);
            entry.second = std::max(entry.second, address + size);
            return true;
        }
        return false;
    };

    for (const auto& entry : allRanges) {
        // check if range exists
        if (isOverlapRange(entry.first, entry.second)) {
            continue;
        }
        uniqueRanges.push_back(std::make_pair(entry.first, entry.second));
    }

    return uniqueRanges;
}

void ReservedMemInfo::reserveFunctionRanges(mlir::func::FuncOp netFunc, MemLiveRangeInfo& liveRangeInfo,
                                            ScanResult& scanResult) {
    ValueOrderedSet liveBuffers;

    netFunc.walk([&](mlir::func::CallOp callOp) {
        auto parentExecOp = callOp->getParentOfType<mlir::async::ExecuteOp>();
        VPUX_THROW_UNLESS(parentExecOp != nullptr, "func::CallOp must have async::ExecuteOp parent");

        // 1. Get buffers that need to be reserved
        const auto buffers = liveRangeInfo.getUsedBuffers(parentExecOp);
        liveBuffers.insert(buffers.begin(), buffers.end());

        // 2. Get unique ranges to reserve
        auto ranges = getUniqueRanges(liveBuffers, scanResult);

        // 3. Reserve ranges in function
        auto calleeName = callOp.getCallee();
        _allReservedMemInfo[calleeName][VPU::MemoryKind::DDR] = std::move(ranges);

        // 4. Free buffers
        for (auto& buffer : buffers) {
            if (liveRangeInfo.eraseUser(buffer, parentExecOp) == 0) {
                liveBuffers.erase(buffer);
            }
        }
    });
}

void ReservedMemInfo::reserveModuleRanges(ScanResult& scanResult) {
    // Add reserved ranges on module level. Those reserved resource might be related
    // to handling of additional special features (e.g. DMA HW profiling)
    auto& moduleReservedMemVec = scanResult.moduleReservedMemVec;
    for (auto& calleeAndMemKindMap : _allReservedMemInfo) {
        auto& reservedAddressAndSizeVec = calleeAndMemKindMap.second[VPU::MemoryKind::DDR];
        reservedAddressAndSizeVec.insert(reservedAddressAndSizeVec.end(), moduleReservedMemVec.begin(),
                                         moduleReservedMemVec.end());
    }
}

void ReservedMemInfo::linearizeCallOps(mlir::func::FuncOp netFunc, AsyncDepsInfo& depsInfo) {
    mlir::async::ExecuteOp prevCallExecOp = nullptr;

    netFunc.walk([&](mlir::func::CallOp callOp) {
        auto parentExecOp = callOp->getParentOfType<mlir::async::ExecuteOp>();
        VPUX_THROW_UNLESS(parentExecOp != nullptr, "func::CallOp must have async::ExecuteOp parent");

        if (prevCallExecOp != nullptr) {
            depsInfo.addDependency(prevCallExecOp, parentExecOp);
        }
        prevCallExecOp = parentExecOp;
    });

    depsInfo.updateTokenDependencies();
}

// returns reserved addresses and sizes for func
ReservedMemInfo::MemReservedMap& ReservedMemInfo::getReservedMemInfo(mlir::StringRef funcName) {
    return _allReservedMemInfo[funcName];
}
