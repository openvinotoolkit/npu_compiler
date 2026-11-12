//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPUIP/strategies_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_depth_to_space_dma_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_distributed_ops_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_per_axis_tile_dma_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_permute_dma_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPUIP/impl/unroll_space_to_depth_dma_strategy.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPUIP/counters_strategy.hpp"
#include "vpux/compiler/dialect/VPUIP/interfaces/strategies.hpp"

#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace vpux::VPUIP {
class StrategyFactory60XX : public VPUIP::StrategyFactory {
public:
    std::unique_ptr<IIterativeWalkPassStrategy> getUnrollDepthToSpaceDMAStrategy(mlir::MLIRContext* ctx,
                                                                                 int64_t dmaPortCount) override {
        return std::make_unique<arch40xx::UnrollDepthToSpaceDMAStrategy>(ctx, dmaPortCount);
    }

    std::unique_ptr<IIterativeWalkPassStrategy> getUnrollSpaceToDepthDMAStrategy(mlir::MLIRContext* ctx,
                                                                                 int64_t dmaPortCount) override {
        return std::make_unique<arch40xx::UnrollSpaceToDepthDMAStrategy>(ctx, dmaPortCount);
    }

    std::unique_ptr<IIterativeWalkPassStrategy> getUnrollPermuteDMAStrategy(mlir::MLIRContext* ctx,
                                                                            int64_t dmaPortCount) override {
        return std::make_unique<arch40xx::UnrollPermuteDMAStrategy>(ctx, dmaPortCount);
    }

    std::unique_ptr<IGreedilyPassStrategy> getUnrollPerAxisTileDMAStrategy(int64_t dmaPortCount) override {
        return std::make_unique<arch40xx::UnrollPerAxisTileDMAStrategy>(dmaPortCount);
    }

    std::unique_ptr<IUnrollDistributedOpsStrategy> getUnrollDistributedOpsStrategy(
            mlir::func::FuncOp funcOp, std::optional<bool> enableSegmentedDmaFusion) override {
        return std::make_unique<arch40xx::UnrollDistributedOpsStrategy>(funcOp, enableSegmentedDmaFusion);
    }

    std::unique_ptr<ICountersStrategy> getCountersStrategy() override {
        return std::make_unique<arch60xx::CountersStrategy>();
    }
};
}  // namespace vpux::VPUIP

void vpux::VPUIP::StrategiesInitializer60XX::initialize(mlir::MLIRContext* context) {
    auto factory = std::make_unique<VPUIP::StrategyFactory60XX>();
    VPUIP::setVPUIPStrategyFactory(context, std::move(factory));
}
