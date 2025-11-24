//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_distributed_ops_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPUIP/transforms/passes/unroll_distributed_ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes/unroll_distributed_ops.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/dma_fusion_utils.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"

#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace vpux::VPUIP::arch40xx {

void UnrollDistributedOpsStrategy::prepareOps(mlir::MLIRContext& ctx, Logger& log) {
    auto module = _funcOp->getParentOfType<mlir::ModuleOp>();
    auto dmaOpExecutor = config::getAvailableExecutor(module, VPU::ExecutorKind::DMA_NN);
    auto dmaPortCount = dmaOpExecutor.getCount();

    const VPUIP::arch37xx::ClusterSWRewriter swRewriter(&ctx, module, log);

    int fusionIdCounter = 0;
    std::optional<VPUIP::DmaFusionHandlerType> dmaFusionHandler;
    if (_enableSegmentedDmaFusion.value_or(false)) {
        dmaFusionHandler = [&](SmallVector<std::pair<VPUIP::NNDMAOp, bool>> fusionInfo) {
            // For 40XX fusion is beneficial only for 4 and 5 clusters because it generates 2 fused DMAs which can load
            // 2 DMA ports. <4 creates only one and >5 creates 3, which makes worse
            constexpr size_t MINIMAL_BENEFICIAL_CLUSTERS = 3;  // In case of 3 clusters we can avoid extra-overhead by
                                                               // fusion of DMAs, while for 2 we leave one port unused
            constexpr size_t MAXIMAL_BENEFICIAL_CLUSTER_INDEX =
                    4;  // 5 cluster case is still benificial because of combination of fusion and split, however 6
                        // cluster fusion will create 3 DMAs for 2 ports
            if (fusionInfo.size() < MINIMAL_BENEFICIAL_CLUSTERS) {
                return;
            }
            bool allCanBeFused = true;
            for (size_t i = 0; i < fusionInfo.size(); i += 2) {
                if (i >= MAXIMAL_BENEFICIAL_CLUSTER_INDEX || i + 1 >= fusionInfo.size()) {
                    break;
                }
                auto curDma = fusionInfo[i].first;
                auto nextDma = fusionInfo[i + 1].first;
                // For 40XX we can fuse only consecutive DMAs because of HW restrictions
                if (fusionInfo[i].second && VPUIP::hasCompatibleTypes(curDma, nextDma)) {
                    // FusionId is also used to assign port and in 3 cluster case we have only one pair, which gives us
                    // not optimial port assignment. To avoid it, we assign fusionId opposite to last DMA port in module
                    // of dmaPort(which is 2)
                    const bool requiresPortAlignment =
                            fusionInfo.size() == 3 &&
                            fusionIdCounter % dmaPortCount == fusionInfo.back().first.getPort().value_or(0);
                    if (requiresPortAlignment) {
                        ++fusionIdCounter;
                    }
                    curDma.setFusionId(fusionIdCounter);
                    nextDma.setFusionId(fusionIdCounter);
                    fusionIdCounter++;
                } else {
                    allCanBeFused = false;
                }
            }
            // For perf gain we need to be sure that we load both ports, otherwise one of ports is idle
            if (!allCanBeFused) {
                for (size_t i = 0; i < fusionInfo.size(); i++) {
                    auto curDma = fusionInfo[i].first;
                    curDma.setFusionIdAttr(nullptr);
                }
            } else {
                // For 3T case fusion has higher benefit than fusion
                if (fusionInfo.size() == 3) {
                    fusionInfo.back().first.setSplitCandidate(false);
                }
            }
        };
    }

    VPUIP::unrollDistributedOpsCommon40XXPlus(_funcOp, std::move(dmaFusionHandler), log);

    _funcOp->walk<mlir::WalkOrder::PostOrder>([&](VPURT::TaskOp vpurtTask) {
        auto op = vpurtTask.getInnerTaskOp();
        if (op == nullptr) {
            return;
        }

        mlir::OpBuilder builder(op);
        if (auto swOp = mlir::dyn_cast<VPUIP::SwKernelOp>(op)) {
            swRewriter.matchAndRewrite(swOp, builder);
        }
    });
}

}  // namespace vpux::VPUIP::arch40xx
