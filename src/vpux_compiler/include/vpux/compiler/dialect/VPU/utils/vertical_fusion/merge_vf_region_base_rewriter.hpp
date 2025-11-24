//
// Copyright (C) 2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/dialect/VPU/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/internal.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_scheduler_interface.hpp"
#include "vpux/compiler/dialect/VPU/utils/vertical_fusion/vertical_fusion_utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

//
// MergeVFRegionRewriter
//

namespace vpux {
namespace VPU {

// set VF users for correct tensor size calculation
class VFSubgraphUserSetter {
public:
    VFSubgraphUserSetter(VerticalFusionOp original, VerticalFusionOp candidate)
            : _mOriginalSubgraph(original), _mCandidateSubgraph(candidate) {
        moveUsers(_mOriginalSubgraph, _mCandidateSubgraph);
    }
    VFSubgraphUserSetter(const VFSubgraphUserSetter&) = delete;

    VFSubgraphUserSetter(VFSubgraphUserSetter&&) = delete;

    VFSubgraphUserSetter& operator=(const VFSubgraphUserSetter&) = delete;

    VFSubgraphUserSetter& operator=(VFSubgraphUserSetter&&) = delete;

    ~VFSubgraphUserSetter() {
        moveUsers(_mCandidateSubgraph, _mOriginalSubgraph);
    }

private:
    void moveUsers(VerticalFusionOp from, VerticalFusionOp to) {
        from.getResult(0).replaceAllUsesWith(to.getResult(0));
    }

    VerticalFusionOp _mOriginalSubgraph;
    VerticalFusionOp _mCandidateSubgraph;
};

template <typename VFCaseType>
class MergeVFRegionBaseRewriter : public mlir::OpRewritePattern<VPU::VerticalFusionOp> {
public:
    using VFConfigType = typename VFCaseType::VFConfigType;
    using IVFSchedulingPtr = std::shared_ptr<IVFScheduling<VFConfigType>>;

    MergeVFRegionBaseRewriter(mlir::MLIRContext* ctx, bool enableVerticalFusionPipelining, bool enablePrefetchTiling,
                              const std::unique_ptr<VPU::LayerVPUNNCost>& costFunction, Logger log)
            : mlir::OpRewritePattern<VPU::VerticalFusionOp>(ctx),
              _enableVerticalFusionPipelining(enableVerticalFusionPipelining),
              _enablePrefetchTiling(enablePrefetchTiling),
              _vpunnCostFunction(costFunction),
              _log(log) {
    }

protected:
    virtual StrategyCost extractVFCost(VFConfigType& vfConfig) const = 0;
    virtual bool canMergeVFOpsWithoutCostCheck(VFCaseType& mergedCase) const = 0;
    virtual bool canSkipMergeVF(VFConfigType& vfConfig, bool opsNeedTiling) const = 0;
    virtual IVFSchedulingPtr detectScenario(VFConfigType& vfConfig) const = 0;
    virtual std::optional<VFCaseType> findVFTiling(VPU::VerticalFusionOp mergedOp, VPU::VerticalFusionOp prevOp,
                                                   VPU::VerticalFusionOp currentOp) const = 0;
    bool checkOtherVFInput(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    bool checkVFCostFunction(VPU::VerticalFusionOp prevOp, VPU::VerticalFusionOp currentOp,
                             VFCaseType& mergedCase) const;
    bool waitOtherUsers(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp) const;
    std::optional<VFCaseType> findVFCase(VPU::VerticalFusionOp newBlock, VPU::VerticalFusionOp parentVFOp,
                                         VPU::VerticalFusionOp mergedVFOp) const;
    bool alignMCTiling(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;
    void fuseBlocks(mlir::PatternRewriter& rewriter, VPU::VerticalFusionOp currentOp,
                    VPU::VerticalFusionOp mergedOp) const;

    VPUNNCostParameters fillInCostParam(mlir::Operation* operation, const OutputTiling& tiling,
                                        const SmallVector<TileInfo>& inputTiles, const bool enablePrefetching) const;

    bool isTileOverOutputChannel(VFConfigType& vfConfig) const;
    bool hasTiling(ArrayRef<int64_t> tilingInfo) const;
    size_t getLinkNumber(VPU::VerticalFusionOp currentOp, VPU::VerticalFusionOp prevOp) const;

    bool _enableVerticalFusionPipelining = false;
    bool _enablePrefetchTiling = true;
    const std::unique_ptr<VPU::LayerVPUNNCost>& _vpunnCostFunction;
    Logger _log;
};
}  // namespace VPU
}  // namespace vpux
