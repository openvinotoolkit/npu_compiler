//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/ops.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIPDPU/transforms/passes/expand_dpu_config/expand_dpu_config_variant.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/utils/attributes.hpp"

namespace vpux::VPUIPDPU::arch40xx::ODU {

mlir::LogicalResult buildODUOutSubtensor(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                         const SmallVector<int64_t>&& start, const SmallVector<int64_t>&& end) {
    if (start.size() != 3 || end.size() != 3) {
        log.error("ODU out subtensor start/end coordinates not properly specified: expected 3 start/end coordinates; "
                  "actual {0}/{1}",
                  start.size(), end.size());
        return mlir::failure();
    }

    builder.create<ODUOutSubtensorOp>(loc, start[0], start[1], start[2], end[0], end[1], end[2]);

    return mlir::success();
}

mlir::LogicalResult buildODUHaloRegionOp(mlir::OpBuilder& builder, const mlir::Location& loc, const Logger& log,
                                         mlir::Block* varBlock, mlir::ArrayAttr haloRegions, bool outSparsityEnabled) {
    if (!haloRegions.empty()) {
        builder.setInsertionPointToEnd(varBlock);

        auto haloCfgOp = builder.create<ODUHaloCfgOp>(loc);
        auto& region = haloCfgOp.getOperation()->getRegion(0);
        auto entryBlock = builder.createBlock(&region);

        if (entryBlock == nullptr) {
            log.error("Error creating entry block for ODUHaloCfgOp");
            return mlir::failure();
        }
    }
    for (const auto& attr : haloRegions) {
        auto haloRegion = mlir::dyn_cast<vpux::VPUIP::DPUHaloRegionAttr>(attr);
        if (!haloRegion) {
            log.error("Got non DPUHaloRegion attribute '{0}' in array", haloRegion);
            return mlir::failure();
        }

        if (haloRegion.getSparsityOffset() && !outSparsityEnabled) {
            log.error("HaloRegion '{0}': sparsity_offset should only exist in case an output sparsity map "
                      "buffer is defined",
                      haloRegion);
            return mlir::failure();
        }

        const auto targetClusters = parseIntArrayAttr<uint8_t>(haloRegion.getTargetClusters());
        uint64_t tileMask = 0;
        for (const auto clusterIdx : targetClusters) {
            tileMask |= 1ull << clusterIdx;
        }
        auto dpuTiles = symbolizeDPUTiles(tileMask);
        if (!dpuTiles.has_value()) {
            log.error("Incorrect target clusters configuration in DPUHaloRegion: {0}", tileMask);
            return mlir::failure();
        }
        builder.create<ODUHaloRegionOp>(loc, haloRegion.getXStart(), haloRegion.getYStart(), haloRegion.getXEnd(),
                                        haloRegion.getYEnd(), haloRegion.getTargetOffset(),
                                        haloRegion.getSparsityOffset(), haloRegion.getTargetWidth(),
                                        DPUTilesAttr::get(builder.getContext(), dpuTiles.value()));
    }

    return mlir::success();
}

}  // namespace vpux::VPUIPDPU::arch40xx::ODU

using namespace vpux::VPUIPDPU::arch40xx::ODU;

mlir::LogicalResult vpux::VPUIPDPU::arch40xx::buildDPUVariantODU(VPUASM::DPUVariantOp origVarOp,
                                                                 mlir::OpBuilder& builder, const Logger& log,
                                                                 mlir::Block* varBlock,
                                                                 ELF::SymbolReferenceMap& symRefMap) {
    if (buildODUOutSubtensor(builder, origVarOp.getLoc(), log, parseIntArrayAttr<int64_t>(origVarOp.getStart()),
                             parseIntArrayAttr<int64_t>(origVarOp.getEnd()))
                .failed()) {
        return mlir::failure();
    }

    auto origInvOp = mlir::cast<VPUASM::DPUInvariantOp>(symRefMap.lookupSymbol(origVarOp.getInvariant()));
    if (auto haloRegions = origVarOp.getHaloRegionsAttr()) {
        if (buildODUHaloRegionOp(builder, origVarOp.getLoc(), log, varBlock, haloRegions,
                                 origInvOp.getOutputSparsityMap().has_value())
                    .failed()) {
            return mlir::failure();
        }
    }

    return mlir::success();
}
