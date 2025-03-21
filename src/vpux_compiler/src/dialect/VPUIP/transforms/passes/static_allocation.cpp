//
// Copyright (C) 2022-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/resources.hpp"

#include "vpux/compiler/core/control_edge_generator.hpp"
#include "vpux/compiler/core/feasible_scheduler_utils.hpp"

#include "vpux/compiler/core/allocation_info.hpp"
#include "vpux/compiler/core/async_deps_info.hpp"
#include "vpux/compiler/core/cost_model_utils.hpp"
#include "vpux/compiler/core/feasible_memory_scheduler_control_edges.hpp"
#include "vpux/compiler/core/linear_scan_handler.hpp"
#include "vpux/compiler/core/mem_live_range_info.hpp"
#include "vpux/compiler/core/reserved_memory_info.hpp"
#include "vpux/compiler/dialect/IE/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPURT/IR/ops.hpp"
#include "vpux/compiler/utils/analysis.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/linear_scan.hpp"

#include "vpux/utils/core/checked_cast.hpp"
#include "vpux/utils/core/error.hpp"

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Value.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPUIP {
#define GEN_PASS_DECL_STATICALLOCATION
#define GEN_PASS_DEF_STATICALLOCATION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

using LinearScanImpl = LinearScan<mlir::Value, LinearScanHandler>;

//
// AllocRewrite
//

template <class ConcreteAllocOp>
class AllocRewrite final : public mlir::OpRewritePattern<ConcreteAllocOp> {
public:
    AllocRewrite(const LinearScanHandler& allocInfo, mlir::MLIRContext* ctx, Logger log)
            : mlir::OpRewritePattern<ConcreteAllocOp>(ctx), _allocInfo(allocInfo), _log(log) {
    }

public:
    mlir::LogicalResult matchAndRewrite(ConcreteAllocOp origOp, mlir::PatternRewriter& rewriter) const final;

private:
    const LinearScanHandler& _allocInfo;
    Logger _log;
};

template <class ConcreteAllocOp>
void createAllocOp(mlir::PatternRewriter&, ConcreteAllocOp, mlir::Type, int64_t) {
    VPUX_THROW("Unsupported allocation operation type");
}

template <>
void createAllocOp(mlir::PatternRewriter& rewriter, mlir::memref::AllocOp origOp, mlir::Type type, int64_t offset) {
    rewriter.replaceOpWithNewOp<VPUIP::StaticAllocOp>(origOp, type, offset);
}

template <>
void createAllocOp(mlir::PatternRewriter& rewriter, VPURT::Alloc origOp, mlir::Type type, int64_t offset) {
    auto section = VPURT::getBufferSection(type.cast<vpux::NDTypeInterface>().getMemoryKind());
    rewriter.replaceOpWithNewOp<VPURT::DeclareBufferOp>(origOp, type, section, nullptr, offset,
                                                        origOp.getSwizzlingKeyAttr());
}

template <class ConcreteAllocOp>
mlir::LogicalResult AllocRewrite<ConcreteAllocOp>::matchAndRewrite(ConcreteAllocOp origOp,
                                                                   mlir::PatternRewriter& rewriter) const {
    _log.trace("Found Alloc Operation '{0}' - '{1}'", origOp->getName(), origOp->getLoc());

    const auto val = origOp->getResult(0);

    for (auto* user : origOp->getUsers()) {
        if (auto iface = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(user)) {
            if (iface.template getEffectOnValue<mlir::MemoryEffects::Free>(val)) {
                return errorAt(origOp, "IR with explicit deallocation operations is not supported");
            }
        }
    }

    const auto offset = checked_cast<int64_t>(_allocInfo.getAddress(val));

    _log.trace("Replace with statically allocated VPURT.DeclareBufferOp (offset = {0})", offset);
    createAllocOp(rewriter, origOp, val.getType(), offset);

    return mlir::success();
}

//
// StaticAllocationPass
//

class StaticAllocationPass final : public VPUIP::impl::StaticAllocationBase<StaticAllocationPass> {
public:
    StaticAllocationPass(VPUIP::MemKindCreateFunc memKindCb, Logger log);

public:
    mlir::LogicalResult initialize(mlir::MLIRContext* ctx) final;

private:
    void safeRunOnFunc() final;

    LinearScanHandler runLinearScan(mlir::func::FuncOp funcOp);

private:
    VPUIP::MemKindCreateFunc _memKindCb;
    VPU::MemoryKind _memKind{VPU::MemoryKind::DDR};
    mlir::SymbolRefAttr _memKindAttr = nullptr;
};

StaticAllocationPass::StaticAllocationPass(VPUIP::MemKindCreateFunc memKindCb, Logger log)
        : _memKindCb(std::move(memKindCb)) {
    Base::initLogger(log, Base::getArgumentName());
}

mlir::LogicalResult StaticAllocationPass::initialize(mlir::MLIRContext* ctx) {
    if (mlir::failed(Base::initialize(ctx))) {
        return mlir::failure();
    }

    auto maybeMemKind = _memKindCb(memSpaceName.getValue());
    if (!maybeMemKind.has_value()) {
        return mlir::failure();
    }

    _memKind = maybeMemKind.value();
    _memKindAttr = mlir::SymbolRefAttr::get(ctx, stringifyEnum(_memKind));

    return mlir::success();
}

LinearScanHandler StaticAllocationPass::runLinearScan(mlir::func::FuncOp funcOp) {
    // A cached deps analysis will be received, if any
    auto& depsInfo = getAnalysis<AsyncDepsInfo>();

    ReservedMemInfo::ReservedAddressAndSizeVector reservedMem;

    // We don't know if we are processing the "main" function here or not (and we don't want to know)
    // Therefore, we want to get the result only if it was prepared and cached using a special module pass
    // in order to avoid a race condition when analyzing and modifying the "main" function.
    auto maybeReservedMemInfo = getCachedParentAnalysis<ReservedMemInfo>();
    if (maybeReservedMemInfo.has_value()) {
        auto& reservedMemInfo = maybeReservedMemInfo.value().get();
        reservedMem = reservedMemInfo.getReservedMemInfo(funcOp.getSymName())[_memKind];
    } else {
        auto module = funcOp->getParentOfType<mlir::ModuleOp>();

        // Check for reserved memory which memory scheduler should take into account
        // so that they not overlap with other buffers. Those reserved resource might be related
        // to handling of additional special features (e.g. DMA HW profiling)
        reservedMem = IE::getReservedMemOffsetAndSizeVec(module, _memKindAttr);
    }

    LinearScanHandler scanHandler;
    std::list<ScheduledOpOneResource> scheduledOpsResources;
    std::optional<ScanResult> cachedScanResult;

    // An empty reserved memory container means that we can use the default scan result for the current function
    // In single-function mode (general approach), the result is cached by calculating the ReservedMemInfo analysis
    if (reservedMem.empty()) {
        // There is no need to call getCachedAnalysis since a cached allocation analysis will be received, if any
        // It is safe to keep references to the resulting objects because the analysis is cached
        auto& allocationInfo = getAnalysis<AllocationInfo>();
        if (allocationInfo.hasResult(_memKind)) {
            cachedScanResult.emplace(allocationInfo.getScanResult(_memKind));
        }
    }

    if (cachedScanResult.has_value()) {
        auto& scanResult = cachedScanResult.value();
        // From MLIR documentation: all analyses are assumed to be invalidated by a pass.
        // So let's just move the instances from the analysis.
        scanHandler = std::move(scanResult.linearScanHandler);
        scheduledOpsResources = std::move(scanResult.scheduledOpOneResource);
    } else {
        auto getMemLiveRangeInfoMemType = [&](VPU::MemoryKind memKind) -> MemLiveRangeInfo& {
            switch (memKind) {
            case VPU::MemoryKind::CMX_NN:
                return getAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::CMX_NN>>();
            case VPU::MemoryKind::DDR:
                return getAnalysis<MemLiveRangeInfoMemType<VPU::MemoryKind::DDR>>();
            default:
                VPUX_THROW("Unsupported memory space: {0}", memKind);
            }
        };
        // A cached deps analysis will be received, if any
        auto& liveRangeInfo = getMemLiveRangeInfoMemType(_memKind);
        // Run a linear scan, giving that a certain amount of memory is reserved
        std::tie(scanHandler, scheduledOpsResources) =
                vpux::runLinearScan(funcOp, liveRangeInfo, depsInfo, _memKind, _log, reservedMem);
    }

    ControlEdgeSet controlEdges;
    ControlEdgeGenerator controlEdgeGenerator;
    // Generate control edges for overlapping memory regions
    controlEdgeGenerator.generateControlEdges(scheduledOpsResources.begin(), scheduledOpsResources.end(), controlEdges);

    _log.trace("Insert control edges for overlapping memory resources");
    _log = _log.nest();

    // Apply dependencies from controlEdges set into depsInfo.
    updateControlEdgesInDepsInfo(depsInfo, controlEdges, _log);

    _log = _log.unnest();

    // Transfer dependencies into tokens between AsyncExecuteOps
    depsInfo.updateTokenDependencies();

    return scanHandler;
}

void StaticAllocationPass::safeRunOnFunc() {
    auto& ctx = getContext();
    auto func = getOperation();

    const auto allocInfo = runLinearScan(func);

    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<VPUIP::VPUIPDialect>();
    target.addLegalDialect<VPURT::VPURTDialect>();
    target.addDynamicallyLegalOp<mlir::memref::AllocOp>([&](mlir::memref::AllocOp op) {
        const auto type = op.getMemref().getType().dyn_cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });
    target.addDynamicallyLegalOp<VPURT::Alloc>([&](VPURT::Alloc op) {
        const auto type = op.getBuffer().getType().dyn_cast<vpux::NDTypeInterface>();
        return type == nullptr || type.getMemoryKind() != _memKind;
    });

    mlir::RewritePatternSet patterns(&ctx);
    patterns.add<AllocRewrite<mlir::memref::AllocOp>>(allocInfo, &ctx, _log);
    patterns.add<AllocRewrite<VPURT::Alloc>>(allocInfo, &ctx, _log);

    if (mlir::failed(mlir::applyPartialConversion(func, target, std::move(patterns)))) {
        _log.error("Failed to replace Alloc/Dealloc Operations");
        signalPassFailure();
    }

    for (auto execOp : func.getOps<mlir::async::ExecuteOp>()) {
        auto* bodyBlock = execOp.getBody();

        for (auto& op : bodyBlock->getOperations()) {
            // Distributed operations are skipped
            if (mlir::isa<VPUIP::NCEClusterTilingOp>(op)) {
                continue;
            }

            if (auto dmaOp = mlir::dyn_cast<VPUIP::DMATypeOpInterface>(op)) {
                // If the port of DMA operation is not initialized, it is set to 0
                if (!dmaOp.getPortVal().has_value()) {
                    auto zeroAttr = vpux::getIntAttr(&getContext(), 0);
                    dmaOp.setPortAttribute(zeroAttr);
                    _log.trace("Uninitialized DMA port at '{0}' set to 0", dmaOp->getLoc());
                }
            }
        }
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::VPUIP::createStaticAllocationPass(MemKindCreateFunc memKindCb, Logger log) {
    return std::make_unique<StaticAllocationPass>(std::move(memKindCb), log);
}
