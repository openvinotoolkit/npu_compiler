//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpux/compiler/core/interfaces/rewriter_pattern_strategies.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/dpu.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/convert_IE_to_VPU_NCE_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/mc_strategy_getter.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/workload_size_constraint.hpp"

namespace vpux {
class NDTypeInterface;
}  // namespace vpux

namespace vpux::VPU {

class StrategyFactory {
public:
    virtual ~StrategyFactory() = default;

    virtual std::unique_ptr<IGreedilyPassStrategy> getMakeOpsWithDistributedTensorStrategy(
            const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
            bool enableExplicitDistributionInfoAttr) = 0;
    virtual std::unique_ptr<StrategyGetterBase> getMultiClusterStrategy(const int64_t numClusters) = 0;
    virtual bool isSmallKernelOptimizationSupported(mlir::Operation* op, const int64_t KX, const int64_t KY,
                                                    const int64_t SX, ArrayRef<VPU::DPUWorkloadOp> workloads) = 0;
    virtual bool doesWorkloadSupportSmallKernelOpt(const int64_t KX, const int64_t KY, const int64_t SX,
                                                   const int64_t padLeft, ArrayRef<int64_t> workloadOutSz,
                                                   bool isFp16Input) = 0;
    virtual bool doesOpSupportSmallKernelOpt(VPU::NCEOpInterface nceOp, const int64_t KX, const int64_t KY,
                                             const int64_t SX) = 0;
    virtual VPU::WorkloadSizeConstraint getWorkloadSizeConstraint() = 0;
    virtual VPU::SparsityConstraint getSparsityConstraint() = 0;

    /**
     * @brief Returns the array of supported channel sizes for Depthwise operations based on the architecture
     *
     * @return ArrayRef<int64_t> Array of supported channel sizes in descending order
     */
    virtual ArrayRef<int64_t> getSupportedChannelsDW() = 0;
    /**
     * @brief Returns a vector of channel sizes that can be optimized
     *
     * @param workloadsChannels Array of available channel configurations
     * @param maxSlotsSum Maximum slots sum constraint for optimization
     * @return SmallVector<int64_t> Vector of channel sizes that can be used for optimized workload split
     */
    virtual SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels,
                                                                          int64_t maxSlotsSum) = 0;
    /**
     * @brief Determines if NCE permute offsets correction is needed
     *
     * @return bool True if NCE permute offsets correction is required, false otherwise
     */
    virtual bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp) = 0;

    // IE to VPU conversion
    virtual std::unique_ptr<IConvertIEToVPUNCEStrategy> getConvertIEToVPUNCEStrategy(const Logger& log) = 0;
};

class StrategyFactoryCache final : public mlir::DialectInterface::Base<StrategyFactoryCache> {
    std::unique_ptr<StrategyFactory> _strategyFactory = nullptr;

public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrategyFactoryCache)

    StrategyFactoryCache(mlir::Dialect* dialect): Base(dialect) {
    }

    const std::unique_ptr<VPU::StrategyFactory>& getStrategyFactory() const {
        return _strategyFactory;
    }

    void setStrategyFactory(std::unique_ptr<VPU::StrategyFactory> strategyFactory) {
        _strategyFactory = std::move(strategyFactory);
    }
};

void setVPUStrategyFactory(mlir::MLIRContext* context, std::unique_ptr<VPU::StrategyFactory> factory);
const std::unique_ptr<VPU::StrategyFactory>& getVPUStrategyFactory(mlir::MLIRContext* context);

}  // namespace vpux::VPU
