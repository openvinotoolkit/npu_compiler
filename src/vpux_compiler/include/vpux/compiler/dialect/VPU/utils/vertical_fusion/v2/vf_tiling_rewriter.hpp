//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_config.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_scheduling_factory.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/v2/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vf_tiling_base_rewriter.hpp"
#include "vpux/compiler/dialect/const/dialect.hpp"

namespace vpux::VPU::VF::v2 {

//
// VerticalFusionTilingRewriter
//

typedef std::function<void(int64_t, mlir::Operation*, mlir::Value&, Shape&)> TilingFunction;

class VerticalFusionTilingRewriter : public VerticalFusionTilingRewriterBase<VFConfig, VFSchedulingFactory> {
public:
    VerticalFusionTilingRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining,
                                 const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : VerticalFusionTilingRewriterBase<VFConfig, VFSchedulingFactory>(ctx, enableVerticalFusionPipelining,
                                                                              costFunction, log) {
    }

protected:
    std::pair<DimArr, int64_t> getDimsData(ArrayRef<int64_t> strategy) const override {
        int64_t tilesLen = 1;
        DimArr dims;
        for (auto item : strategy | indexed) {
            auto dim = Dim(item.index());
            auto tileSize = item.value();
            if (tileSize > 1) {
                dims.push_back(dim);
                tilesLen *= tileSize;
            }
        }
        return std::make_pair(dims, tilesLen);
    }

    virtual TilingStorage restoreTilingStorage(VFConfig& config, ArrayRef<int64_t> strategy,
                                               TilingOperationStorage::UPtr& operationStorage) const override {
        auto storage = calculateTilingRegions(config, strategy, _log, operationStorage);

        VPUX_THROW_WHEN(mlir::failed(storage), "Cannot restore tiling regions for VF {0}", config.getSubgraph());

        return storage.value();
    }
};
}  // namespace vpux::VPU::VF::v2
