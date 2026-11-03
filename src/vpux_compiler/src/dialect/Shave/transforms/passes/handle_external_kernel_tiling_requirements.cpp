//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"
#include "vpux/compiler/dialect/Shave/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/utils/attributes.hpp"
#include "vpux/compiler/utils/passes.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux::Shave {
#define GEN_PASS_DECL_HANDLEEXTERNALKERNELTILINGREQUIREMENTS
#define GEN_PASS_DEF_HANDLEEXTERNALKERNELTILINGREQUIREMENTS
#include "vpux/compiler/dialect/Shave/passes.hpp.inc"
}  // namespace vpux::Shave

using namespace vpux;

namespace {

//
// HandleExternalKernelTilingRequirements
//

// Returns the ScratchLocation for a scratch operand given the op's
// scratch_locations ArrayAttr.  The scratch operands follow the $inputs group,
// so scratchIdx = operandIdx - numInputs.
static std::optional<Shave::ScratchLocation> scratchLocationAt(Shave::ExternalKernelOp op, unsigned operandIdx) {
    auto locAttr = op.getScratchLocationsAttr();
    if (!locAttr) {
        return std::nullopt;
    }
    const auto numInputs = static_cast<unsigned>(op.getInputs().size());
    if (operandIdx < numInputs) {
        return std::nullopt;
    }
    const auto scratchIdx = operandIdx - numInputs;
    const auto locations = locAttr.getValue();
    if (scratchIdx >= locations.size()) {
        return std::nullopt;
    }
    return mlir::cast<Shave::ScratchLocationAttr>(locations[scratchIdx]).getValue();
}

// Converts a single ExternalKernelOp to RepeatedExternalKernelOp.
// Precondition: MaterializeScratchBuffers has already run so
// getScratchBuffers() is non-empty and getScratchLocationsAttr() is set.
static void convertToRepeated(Shave::ExternalKernelOp extKernelOp, int64_t numShaves, int64_t numTiles, Logger& log) {
    auto* ctx = extKernelOp->getContext();
    auto loc = extKernelOp->getLoc();

    // Partition scratch_buffers into CMX-pinned and DDR/AUTO groups.
    // For REPEATED mode:
    //   CMX  -> cmx_scratch_buffers (will be DUPLICATED per tile)
    //   DDR  -> ddr_scratch_buffers (shared across tiles)
    //   AUTO -> ddr_scratch_buffers (conservative: shared DDR for REPEATED)
    SmallVector<mlir::Value> cmxScratch, ddrScratch;
    for (auto& operand : extKernelOp.getScratchBuffersMutable()) {
        auto locOpt = scratchLocationAt(extKernelOp, operand.getOperandNumber());
        if (locOpt && *locOpt == Shave::ScratchLocation::CMX) {
            cmxScratch.push_back(operand.get());
        } else {
            ddrScratch.push_back(operand.get());
        }
    }

    log.debug("Converting ExternalKernelOp '{0}' -> RepeatedExternalKernelOp "
              "({1} CMX scratch, {2} DDR scratch, numShaves={3}, numTiles={4})",
              extKernelOp.getUniqueId(), cmxScratch.size(), ddrScratch.size(), numShaves, numTiles);

    auto distAttr = Shave::KernelDistributionAttr::get(ctx, getIntAttr(ctx, numShaves), getIntAttr(ctx, numTiles));

    mlir::OpBuilder builder(extKernelOp);

    auto newOp = Shave::RepeatedExternalKernelOp::create(
            builder, loc, extKernelOp.getResultTypes(), extKernelOp.getInputs(), cmxScratch, ddrScratch,
            extKernelOp.getConfigAttr(), extKernelOp.getKernelAttributesAttr(), extKernelOp.getUniqueIdAttr(),
            extKernelOp.getUsesDmaAttr(), extKernelOp.getUsesDpuAttr(), distAttr,
            /*multiClusterStrategy=*/nullptr);

    extKernelOp.replaceAllUsesWith(newOp.getResults());
    extKernelOp.erase();
}

//
// Pass
//

class HandleExternalKernelTilingRequirements final :
        public Shave::impl::HandleExternalKernelTilingRequirementsBase<HandleExternalKernelTilingRequirements> {
public:
    explicit HandleExternalKernelTilingRequirements(Logger log): _log(std::move(log)) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void HandleExternalKernelTilingRequirements::safeRunOnFunc() {
    auto func = getOperation();
    auto moduleOp = func->getParentOfType<mlir::ModuleOp>();
    VPUX_THROW_UNLESS(moduleOp, "HandleExternalKernelTilingRequirements: could not find parent ModuleOp");

    const auto numTiles = config::getTileExecutor(moduleOp).getCount();
    const auto numShaves = config::getNumOfEnginesOnTile(moduleOp, config::ExecutorKind::SHAVE_ACT);

    // Collect ops first to avoid iterator invalidation during replacement.
    SmallVector<Shave::ExternalKernelOp> targets;
    func.walk([&](Shave::ExternalKernelOp op) {
        targets.push_back(op);
    });

    for (auto extKernelOp : targets) {
        auto configAttrOpt = extKernelOp.getConfig();
        if (!configAttrOpt.has_value()) {
            continue;
        }

        auto configOp = mlir::dyn_cast_or_null<Shave::CustomOpConfigOp>(
                mlir::SymbolTable::lookupNearestSymbolFrom(extKernelOp, configAttrOpt.value()));
        if (!configOp) {
            continue;
        }

        auto tilingOps = configOp.getOps<Shave::CustomOpTilingConfigOp>();
        if (tilingOps.empty()) {
            continue;
        }

        VPUX_THROW_UNLESS(std::distance(tilingOps.begin(), tilingOps.end()) == 1,
                          "Multiple tiling configurations found for ExternalKernel '{0}'", extKernelOp.getUniqueId());

        auto tilingConfigOp = *tilingOps.begin();
        if (tilingConfigOp.getMode() != Shave::TilingMode::REPEATED) {
            continue;
        }

        VPUX_THROW_UNLESS(extKernelOp.getUsesDma(), "ExternalKernel '{0}' with REPEATED tiling mode must use DMA",
                          extKernelOp.getUniqueId());

        auto scratchLocations = extKernelOp.getScratchLocationValues();
        VPUX_THROW_UNLESS(llvm::any_of(scratchLocations,
                                       [](Shave::ScratchLocation loc) {
                                           return loc == Shave::ScratchLocation::CMX;
                                       }),
                          "ExternalKernel '{0}' with REPEATED tiling mode must have at least one CMX scratch buffer",
                          extKernelOp.getUniqueId());

        convertToRepeated(extKernelOp, numShaves, numTiles, _log);
    }
}

}  // namespace

std::unique_ptr<mlir::Pass> vpux::Shave::createHandleExternalKernelTilingRequirementsPass(Logger log) {
    return std::make_unique<HandleExternalKernelTilingRequirements>(std::move(log));
}
