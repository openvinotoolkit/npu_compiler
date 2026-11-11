//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU60XX/dialect/VPU/strategies_initializer.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/make_ops_with_distributed_tensor_strategy.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/mc_strategy_getter.hpp"
#include "vpux/compiler/NPU40XX/dialect/VPU/impl/sparsity_constraint.hpp"
#include "vpux/compiler/NPU50XX/conversion/impl/convert_IE_to_VPU_NCE_strategy.hpp"
#include "vpux/compiler/NPU50XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/impl/convert_tile_with_eltwise_to_nce_pool_strategy.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/impl/nce_workload_channels.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/impl/small_kernel_optimization.hpp"
#include "vpux/compiler/NPU60XX/dialect/VPU/impl/workload_size_constraint.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/eltwise.hpp"
#include "vpux/compiler/dialect/IE/IR/ops/pooling.hpp"
#include "vpux/compiler/dialect/IE/utils/matmul.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/convolution.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/image.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/strategies.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"

#include <mlir/IR/MLIRContext.h>

using namespace vpux;

namespace vpux::VPU {
class StrategyFactory60XX : public VPU::StrategyFactory {
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
        auto nceOp = mlir::dyn_cast_if_present<VPU::NCEOpInterface>(op);
        if (nceOp == nullptr || !doesOpSupportSmallKernelOpt(nceOp, KX, KY, SX)) {
            return false;
        }

        return VPU::arch60xx::isSmallKernelOptimizationSupported(workloads);
    }

    bool doesWorkloadSupportSmallKernelOpt(const int64_t KX, const int64_t KY, const int64_t SX, const int64_t padLeft,
                                           ArrayRef<int64_t> workloadOutSz, bool isFp16Input) override {
        return VPU::arch50xx::doesWorkloadSupportSmallKernelOpt(KX, SX, workloadOutSz, isFp16Input, KY, padLeft);
    }

    bool doesOpSupportSmallKernelOpt(VPU::NCEOpInterface nceOp, const int64_t KX, const int64_t /*KY*/,
                                     const int64_t SX) override {
        if (!mlir::isa<VPU::NCEDepthConvolutionOp>(nceOp)) {
            return false;
        }

        if (KX == 1 || (SX != 1 && SX != 2)) {
            return false;
        }

        const bool hasSparse = mlir::isa<VPU::SparseTensorType>(nceOp->getOperand(0).getType()) ||
                               mlir::isa<VPU::SparseTensorType>(nceOp->getResult(0).getType());
        if (!hasSparse) {
            return true;
        }

        return VPU::NCEInvariant::isSparseWorkloadEligibleForSmallKernelOpt(nceOp.getOperation(),
                                                                            getSupportedChannelsDW());
    }

    VPU::WorkloadSizeConstraint getWorkloadSizeConstraint() override {
        return VPU::arch60xx::WorkloadSizeConstraint{};
    }

    VPU::SparsityConstraint getSparsityConstraint() override {
        return VPU::arch40xx::SparsityConstraint{};
    }

    ArrayRef<int64_t> getSupportedChannelsDW() override {
        return VPU::arch60xx::supportedChannelsForNPU60XX;
    }

    SmallVector<int64_t> getChannelsSupportedByKernelOptimization(ArrayRef<int64_t> workloadsChannels,
                                                                  int64_t maxSlotsSum) override {
        return VPU::arch60xx::getChannelsSupportedByKernelOptimization(workloadsChannels, maxSlotsSum);
    }

    bool isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface) override {
        return false;
    }

    bool useNewWeightTableFormat(mlir::Operation* op, bool isCompressConv) override {
        return llvm::TypeSwitch<mlir::Operation*, bool>(op)
                // Convolution
                .Case<IE::ConvolutionOp, VPU::NCEConvolutionOp>([&](auto) {
                    return !isCompressConv;
                })
                // GroupConvolution
                .Case<IE::GroupConvolutionOp>([&](IE::GroupConvolutionOp op) {
                    auto bias = op.getBias();
                    return bias != nullptr && bias.getDefiningOp<Const::DeclareOp>() == nullptr;
                })
                // MatMul
                .Case<IE::MatMulOp, VPU::NCEMatMulOp>([](auto) {
                    return true;
                })
                // SEP
                .Case<VPU::TransposedConvolutionOp, VPU::RollOp, VPU::PadOp>([](auto) {
                    return true;
                })
                // Pooling
                .Case<IE::AvgPoolOp, VPU::NCEAveragePoolOp, IE::MaxPoolOp, VPU::NCEMaxPoolOp>([](auto) {
                    return true;
                })
                // Eltwise
                .Case<IE::AddOp, IE::SubtractOp, IE::MultiplyOp, VPU::NCEEltwiseOp>([](auto) {
                    return true;
                })
                .Default([](mlir::Operation*) {
                    return false;
                });
    }

    std::unique_ptr<IIterativeWalkPassStrategy> getConvertTileWithEltwiseToNCEPoolStrategy(
            mlir::MLIRContext* ctx) override {
        return std::make_unique<arch60xx::ConvertTileWithEltwiseToNCEPoolStrategy>(ctx);
    }

    std::unique_ptr<IConvertIEToVPUNCEStrategy> getConvertIEToVPUNCEStrategy(const Logger& log) override {
        return std::make_unique<vpux::arch50xx::ConvertIEToVPUNCEStrategy>(log, config::ArchKind::NPU60XX);
    }
};
}  // namespace vpux::VPU

void vpux::VPU::StrategiesInitializer60XX::initialize(mlir::MLIRContext* context) {
    auto factory = std::make_unique<VPU::StrategyFactory60XX>();
    VPU::setVPUStrategyFactory(context, std::move(factory));
}
