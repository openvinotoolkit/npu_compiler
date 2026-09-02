//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU50XX/dialect/VPU/strategies_initializer.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/NPU37XX/dialect/VPU/impl/workload_size_constraint.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/mc_strategy_getter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/NPU50XX/conversion/impl/convert_IE_to_VPU_NCE_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace vpux::VPU {
class StrategyFactory50XX : public VPU::StrategyFactory {
    std::unique_ptr<IGreedilyPassStrategy> getMakeOpsWithDistributedTensorStrategy(
            const llvm::DenseMap<mlir::OpResult, vpux::NDTypeInterface>& typeLookup,
            const llvm::DenseMap<mlir::Operation*, llvm::DenseMap<int, vpux::NDTypeInterface>>& inputTypeLookup,
            bool) override {
        return std::make_unique<arch40xx::MakeOpsWithDistributedTensorStrategy>(typeLookup, inputTypeLookup);
    }

    std::unique_ptr<StrategyGetterBase> getMultiClusterStrategy(const int64_t numClusters) override {
        if (numClusters == 1) {
            return std::make_unique<StrategyGetterBase>();
        }
        return std::make_unique<arch40xx::StrategyGetter>(numClusters);
    }

    bool isSmallKernelOptimizationSupported(mlir::Operation* op, const int64_t KX, const int64_t KY, const int64_t SX,
                                            ArrayRef<VPU::DPUWorkloadOp> workloads) override {
        // 64 workload_size_z have some additional constraints on NPU5 and we excluded the
        // VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT64 in workload channels splitting because of performance
        // regression, but here are some special cases that can gain benefits.
        const auto hasInt8Act = [&] {
            auto elemType = mlir::cast<vpux::NDTypeInterface>(op->getOperand(0).getType()).getElementType();
            if (auto quantType = mlir::dyn_cast<mlir::quant::QuantizedType>(elemType)) {
                return quantType.getStorageType().isInteger(8);
            }

            return false;
        }();
        const auto allWorkloadsHave64OutputZ = llvm::all_of(workloads, [](auto wl) {
            return wl.getConstOutputSizes()[Dims4D::Act::C.ind()] == VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT64;
        });
        if (hasInt8Act && mlir::isa<VPU::NCEDepthConvolutionOp>(op) && allWorkloadsHave64OutputZ &&
            (KX == 3 && SX == 1)) {
            return true;
        }

        auto nceOp = mlir::dyn_cast_if_present<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr || !doesOpSupportSmallKernelOpt(nceOp, KX, KY, SX)) {
            return false;
        }

        return VPU::arch50xx::isSmallKernelOptimizationSupported(op, KX, KY, workloads);
    }

    bool doesWorkloadSupportSmallKernelOpt(const int64_t KX, const int64_t KY, const int64_t SX, const int64_t padLeft,
                                           ArrayRef<int64_t> workloadOutSz, bool isFp16Input) override {
        return VPU::arch50xx::doesWorkloadSupportSmallKernelOpt(KX, SX, workloadOutSz, isFp16Input, KY, padLeft);
    }

    bool doesOpSupportSmallKernelOpt(VPU::NCEOpInterface nceOp, const int64_t KX, const int64_t /*KY*/,
                                     const int64_t SX) override {
        // config.dw3x3s1OptDisable is set to false for both DepthConv and AveragePool on NPU5
        if (!mlir::isa<VPU::NCEDepthConvolutionOp, VPU::NCEAveragePoolOp>(nceOp)) {
            return false;
        }

        if (!(KX == 3 && SX == 1)) {
            return false;
        }

        const bool hasSparse = mlir::isa<VPU::SparseTensorType>(nceOp->getOperand(0).getType()) ||
                               mlir::isa<VPU::SparseTensorType>(nceOp->getResult(0).getType());
        if (!hasSparse) {
            return true;
        }

        const auto supportedChannels = {VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT16,
                                        VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT32};
        return VPU::NCEInvariant::isSparseWorkloadEligibleForSmallKernelOpt(nceOp.getOperation(), supportedChannels);
    }

    VPU::WorkloadSizeConstraint getWorkloadSizeConstraint() override {
        return VPU::arch37xx::WorkloadSizeConstraint{};
    }

    VPU::SparsityConstraint getSparsityConstraint() override {
        return VPU::arch40xx::SparsityConstraint{};
    }

    ArrayRef<int64_t> getSupportedChannelsDW() override {
        return VPU::arch37xx::supportedChannelsForNPU37XX;
    }

    SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels,
                                                                  int64_t maxSlotsSum) override {
        return VPU::arch40xx::getChannelsSupportedByKernelOptimization(workloadsChannels, maxSlotsSum);
    }

    bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface) override {
        return false;
    }

    bool isConvertTileWithEltwiseToNCEPoolSupported() override {
        return false;
    }

    std::unique_ptr<IConvertIEToVPUNCEStrategy> getConvertIEToVPUNCEStrategy(const Logger& log) override {
        return std::make_unique<vpux::arch50xx::ConvertIEToVPUNCEStrategy>(log, config::ArchKind::NPU50XX);
    }
};
}  // namespace vpux::VPU

void vpux::VPU::StrategiesInitializer50XX::initialize(mlir::MLIRContext* context) {
    auto factory = std::make_unique<VPU::StrategyFactory50XX>();
    VPU::setVPUStrategyFactory(context, std::move(factory));
}
