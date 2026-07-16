//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/utils/dynamic_memref_bounds.hpp"

#include "vpux/compiler/dialect/core/types.hpp"
#include "vpux/compiler/dialect/net/IR/ops.hpp"
#include "vpux/compiler/dialect/net/utils/network_info_utils.hpp"
#include "vpux/compiler/utils/memref_attr_utils.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>

using namespace vpux;

// In the HostCompile + dynamic tensor flow:
// - Compute functions have dynamic shapes (e.g., memref<?x64xf16>)
// - Core.ReinterpretCast makes shapes static inside the function body
// - At call boundaries, outlined kernel args/results must remain dynamic in IR
// - But the physical buffers must be allocated to the declared upper bounds (can be taken from callee NetworkInfo)
//
// This module provides utilities to:
// - Extract upper bounds from callee net::NetworkInfoOp
// - Allocate memrefs with those bounds materialized in the dynamic dimensions
// - Preserve the original dynamic memref type in IR semantics

namespace {

// Resolve the mlir::ModuleOp from a callee symbol reference
// Navigates through the call operation's callable reference to find the target module
mlir::ModuleOp resolveCalleeModule(mlir::CallOpInterface callOp) {
    auto callableRef = callOp.getCallableForCallee();
    auto symbRef = mlir::dyn_cast<mlir::SymbolRefAttr>(callableRef);
    if (!symbRef) {
        return nullptr;
    }

    auto parentModule = callOp->getParentOfType<mlir::ModuleOp>();
    if (!parentModule) {
        return nullptr;
    }

    // Flat symbol references (e.g. func.call @main) resolve to a function in the same module, not
    // to a nested module with its own NetworkInfo. Only Core.NestedCall-style callee refs
    // (@Module::@func) have a module prefix that owns a separate NetworkInfo.
    if (symbRef.getNestedReferences().empty()) {
        return nullptr;
    }

    // For nested symbols, look up the root reference module
    return parentModule.lookupSymbol<mlir::ModuleOp>(symbRef.getRootReference());
}

mlir::MemRefType getAllocatableMemRefType(mlir::MemRefType memrefType) {
    // memref.alloc does not accept strided layouts; strip it so the alloc type is compatible.
    // The caller is responsible for casting back to the original strided type after allocation
    if (!hasStridedLayout(memrefType)) {
        return memrefType;
    }

    return mlir::MemRefType::get(memrefType.getShape(), memrefType.getElementType(), nullptr,
                                 memrefType.getMemorySpace());
}

}  // namespace

// Retrieve the upper bounds for a specific buffer (arg or result) from the callee's net::NetworkInfoOp
SmallVector<int64_t> vpux::VPUIP::getBoundsForCalleeBuffer(mlir::CallOpInterface callOp, size_t bufferIdx, bool isInput,
                                                           Logger& log, StringRef sourceTag) {
    auto calleeModule = resolveCalleeModule(callOp);
    if (!calleeModule) {
        log.trace("[{0}] Can't resolve callee module for buffer idx={1}, isInput={2}", sourceTag, bufferIdx, isInput);
        return {};
    }

    // Locate the single net::NetworkInfoOp in the callee module, which declares the I/O bounds
    auto netOps = to_small_vector(calleeModule.getOps<net::NetworkInfoOp>());
    if (netOps.size() != 1) {
        log.trace("[{0}] Expected one net::NetworkInfoOp in callee module, got {1}", sourceTag, netOps.size());
        return {};
    }

    auto netInfo = netOps.front();
    // DataInfo for profiling output is not a user-visible network buffer, so we only query
    // inputs and outputs here. Profiling bounds are handled separately in the ELF size path
    const auto dataInfos = isInput ? netInfo.getInputsDataInfo() : netInfo.getOutputsDataInfo();
    const auto bounds = net::getBoundsFromDataInfo(dataInfos, bufferIdx);
    if (bounds.empty()) {
        log.trace("[{0}] No BoundedTensorType found in DataInfo[{1}] (isInput={2})", sourceTag, bufferIdx, isInput);
    }
    return bounds;
}

namespace {

mlir::Value allocateDynamic(mlir::CallOpInterface callOp, size_t bufferIdx, bool isInput, mlir::MemRefType memrefType,
                            mlir::MemRefType allocType, mlir::OpBuilder& builder, Logger& log, StringRef sourceTag) {
    auto bounds = VPUIP::getBoundsForCalleeBuffer(callOp, bufferIdx, isInput, log, sourceTag);
    VPUX_THROW_WHEN(bounds.empty(),
                    "[{0}] Cannot allocate dynamic memref '{1}' for {2} buffer {3}: no upper bounds in callee "
                    "net::NetworkInfoOp",
                    sourceTag, memrefType, isInput ? "input" : "output", bufferIdx);

    // Each dynamic dimension needs an arith.constant operand for memref.alloc
    // Static dimensions are encoded in the memref type shape and require no operands
    SmallVector<mlir::Value> dynDimOperands;
    SmallVector<int64_t> usedBounds;
    for (int64_t dimIdx = 0; dimIdx < memrefType.getRank(); ++dimIdx) {
        if (!memrefType.isDynamicDim(dimIdx)) {
            continue;
        }

        VPUX_THROW_UNLESS(dimIdx < static_cast<int64_t>(bounds.size()),
                          "[{0}] Bound array size {1} is smaller than dynamic dim index {2} for type '{3}'", sourceTag,
                          bounds.size(), dimIdx, memrefType);
        usedBounds.push_back(bounds[dimIdx]);
        dynDimOperands.push_back(builder.create<mlir::arith::ConstantIndexOp>(callOp.getLoc(), bounds[dimIdx]));
    }

    log.trace("[{0}] Allocate dynamic memref with upper bounds: idx={1}, isInput={2}, type={3}, usedBounds={4}",
              sourceTag, bufferIdx, isInput, memrefType, usedBounds);

    // The alloc type has the same shape as memrefType but without strided layout, so
    // memref.alloc produces a contiguous buffer. If the original type was strided,
    // a memref.cast restores the layout for the rest of the IR to use.
    auto alloc = builder.create<mlir::memref::AllocOp>(callOp.getLoc(), allocType, dynDimOperands).getMemref();
    if (allocType == memrefType) {
        return alloc;
    }

    return builder.create<mlir::memref::CastOp>(callOp.getLoc(), memrefType, alloc).getResult();
}

}  // namespace

mlir::Value vpux::VPUIP::allocateCallBoundaryMemref(mlir::CallOpInterface callOp, size_t bufferIdx, bool isInput,
                                                    mlir::MemRefType memrefType, mlir::OpBuilder& builder, Logger& log,
                                                    StringRef sourceTag) {
    // Strip strided layout once; passed to both the static alloc and the dynamic helper
    // to avoid recomputing it
    const auto allocType = getAllocatableMemRefType(memrefType);

    if (memrefType.getNumDynamicDims() == 0) {
        // Static memref: allocate directly. If the original type was strided,
        // cast back so callers see the correct layout
        auto alloc = builder.create<mlir::memref::AllocOp>(callOp.getLoc(), allocType).getMemref();
        if (allocType == memrefType) {
            return alloc;
        }

        return builder.create<mlir::memref::CastOp>(callOp.getLoc(), memrefType, alloc).getResult();
    }

    // Dynamic memref: resolve per-dim upper bounds from the callee's NetworkInfo and
    // materialise them as arith.constant operands to memref.alloc so the physical buffer
    // is sized for the worst-case runtime shape
    return allocateDynamic(callOp, bufferIdx, isInput, memrefType, allocType, builder, log, sourceTag);
}
