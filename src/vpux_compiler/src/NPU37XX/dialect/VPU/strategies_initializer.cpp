//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPU/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/conversion/impl/convert_IE_to_VPU_NCE_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/mc_strategy_getter.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/workload_size_constraint.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace vpux::VPU {
class StrategyFactory37XX : public VPU::StrategyFactory {
    std::unique_ptr<IGreedilyPassStrategy> getMakeOpsWithDistributedTensorStrategy(
            const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
            bool enableExplicitDistributionInfoAttr) override {
        return std::make_unique<arch37xx::MakeOpsWithDistributedTensorStrategy>(typeLookup, inputTypeLookup,
                                                                                enableExplicitDistributionInfoAttr);
    }

    std::unique_ptr<StrategyGetterBase> getMultiClusterStrategy(const int64_t numClusters) override {
        if (numClusters == 1) {
            return std::make_unique<StrategyGetterBase>();
        }
        return std::make_unique<arch37xx::StrategyGetter>();
    }

    bool isSmallKernelOptimizationSupported(mlir::Operation* /*op*/, const int64_t /*KX*/, const int64_t /*KY*/,
                                            const int64_t /*SX*/, ArrayRef<VPU::DPUWorkloadOp> /*workloads*/) override {
        // Small kernel optimization is not supported on NPU37XX.
        return false;
    };

    bool doesWorkloadSupportSmallKernelOpt(const int64_t /*KX*/, const int64_t /*KY*/, const int64_t /*SX*/,
                                           const int64_t /*padLeft*/, ArrayRef<int64_t> /*workloadOutSz*/,
                                           bool /*isFp16Input*/) override {
        // Small kernel optimization is not supported on NPU37XX.
        return false;
    }

    bool doesOpSupportSmallKernelOpt(VPU::NCEOpInterface /*nceOp*/, const int64_t /*KX*/, const int64_t /*KY*/,
                                     const int64_t /*SX*/) override {
        // Small kernel optimization is not supported on NPU37XX.
        return false;
    }

    VPU::WorkloadSizeConstraint getWorkloadSizeConstraint() override {
        return VPU::arch37xx::WorkloadSizeConstraint{};
    }

    VPU::SparsityConstraint getSparsityConstraint() override {
        return VPU::arch37xx::SparsityConstraint{};
    }

    ArrayRef<int64_t> getSupportedChannelsDW() override {
        return VPU::arch37xx::supportedChannelsForNPU37XX;
    }

    SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> /*workloadsChannels*/,
                                                                  int64_t /*maxSlotsSum*/) override {
        return {};
    }

    bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface nceOp) override {
        auto outputType = nceOp->getResult(0).getType();
        return mlir::isa<vpux::VPU::DistributedTensorType>(outputType);
    }

    std::unique_ptr<IConvertIEToVPUNCEStrategy> getConvertIEToVPUNCEStrategy(const Logger& log) override {
        return std::make_unique<vpux::arch37xx::ConvertIEToVPUNCEStrategy>(log, config::ArchKind::NPU37XX);
    }
};
}  // namespace vpux::VPU

void vpux::VPU::StrategiesInitializer37XX::initialize(mlir::MLIRContext* context) {
    auto factory = std::make_unique<VPU::StrategyFactory37XX>();
    VPU::setVPUStrategyFactory(context, std::move(factory));
}
