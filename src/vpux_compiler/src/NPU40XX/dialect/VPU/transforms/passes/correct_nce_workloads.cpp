//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/NPU40XX/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/core/layers.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter_base.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_invariant.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"

namespace vpux::VPU::arch40xx {
#define GEN_PASS_DECL_CORRECTNCEWORKLOADS
#define GEN_PASS_DEF_CORRECTNCEWORKLOADS
#include "vpux/compiler/NPU40XX/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU::arch40xx

using namespace vpux;
using namespace VPU;

vpux::VPU::WorkloadSplitter40XX::WorkloadSplitter40XX(mlir::func::FuncOp funcOp, vpux::Logger log)
        : WorkloadSplitterBase(funcOp, vpux::VPU::supportedChannelsDW, log) {
}

SmallVector<Shape> vpux::VPU::WorkloadSplitter40XX::getPerClusterOffsetsCorrection(VPU::NCEOpInterface) {
    return {};
}

bool vpux::VPU::WorkloadSplitter40XX::isNCEPermuteOffsetsCorrectionNeeded(VPU::NCEOpInterface) {
    return false;
}

SmallVector<int64_t> vpux::VPU::WorkloadSplitter40XX::getSupportedChannels(
        const mlir::DenseSet<mlir::Operation*>& nceOps, const VPU::SparsityConstraint& sparsityConstraint) {
    auto supportedChannels = WorkloadSplitterBase::getSupportedChannels(nceOps, sparsityConstraint);

    const auto hasSparseOutput = llvm::any_of(nceOps, [](mlir::Operation* op) {
        return op->getResult(0).getType().isa<VPU::SparseTensorType>();
    });
    if ((std::find(supportedChannels.begin(), supportedChannels.end(), VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT) !=
         supportedChannels.end()) &&
        nceOps.size() == 1 && isDepthwiseOp(*nceOps.begin()) && !hasSparseOutput) {
        auto nceOp = mlir::dyn_cast<VPU::NCEOpInterface>(*nceOps.begin());
        const auto kernelSize = nceOp.getKernelSizeVal();
        const auto KX = kernelSize[Dims4D::Kernel::X.ind()];
        const auto kernelStride = nceOp.getStridesVal();
        const auto SX = kernelStride[Dims4D::Strides::X.ind()];
        const auto outputType = nceOp.getOperation()->getResult(0).getType().cast<NDTypeInterface>();
        const auto OC = outputType.getShape()[vpux::Dims4D::Act::C];

        SmallVector<int64_t> workloadsChannels = {OC};
        // Get a set containing all the channels from the workloads of the given NCE operation if workloads has created
        // in current phase
        auto workloads = nceOp.getWorkloads().getOps<VPU::DPUWorkloadOp>();
        if (!workloads.empty()) {  // Already owns workloads
            workloadsChannels = to_container<SmallVector<int64_t>>(
                    workloads | transformed([](VPU::DPUWorkloadOp workload) -> int64_t {
                        const auto wlSizes = parseIntArrayAttr<int64_t>(workload.getOutSizes());
                        return wlSizes[Dims4D::Act::C.ind()];
                    }));
        } else {  // No workloads splitted
            const auto getPerClusterShapes = [&]() {
                auto clusterOp = mlir::dyn_cast<VPU::ClusteredOpInterface>(*nceOps.begin());
                if (clusterOp == nullptr || !clusterOp.getMultiClusterStrategy().has_value()) {
                    return SmallVector<Shape>{outputType.getShape().raw()};
                }
                // multi cluster case
                auto strategy = clusterOp.getMultiClusterStrategy().value();
                auto numClusters = VPU::getOptimalNumClusters(clusterOp, outputType.getShape(), strategy);
                auto distributedType = mlir::cast<VPU::DistributedTensorType>(
                        getDistributedOutputTypeFromOp(clusterOp, outputType, numClusters, strategy)
                                .getDistributedTypes()
                                .front());
                return distributedType.getPerClusterComputeShapes();
            };

            const auto perClusterShapes = getPerClusterShapes();
            if (!perClusterShapes.empty()) {
                workloadsChannels = to_container<SmallVector<int64_t>>(
                        perClusterShapes | transformed([](ShapeRef clusterShape) -> int64_t {
                            return clusterShape[Dims4D::Act::C];
                        }));
            }
        }

        auto workloadChannelsMeetRequirement = llvm::all_of(workloadsChannels, [&](const auto& channel) {
            return channel % VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT == 0;
        });

        size_t workloadNumInTotal = 0;
        for (auto channel : workloadsChannels) {
            workloadNumInTotal += (channel / VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT);
        }

        // This is a performance opt to use VPU_CHANNEL_SIZE_FOR_L1OPT as supportedChannels on 40XX
        // for DW ops with KX = 3 and SX = 1. Hardware has a specific support for that kind of workloads
        // Note that if the total workload num is greater than the max barrier slot num, which means that the workload
        // will be executed in linearization, so disable the optimization under this case.
        // One more limitation: do not split sparse inputs.
        // If the compiler split 64x16x16 workload, each slice would need to access storage element table.
        // Storage element table access introduces an overhead.
        // There are two valid configurations:
        // 1. seDepth = 2 with 2 * 32x16x16 workloads
        // 2. seDepth = 1 with 1 * 64x16x16 workload
        // The latter is preferable since this way DPU tries to access storage element table for only one workload.
        const auto hasSparseInput = mlir::isa<VPU::SparseTensorType>(nceOp->getOperand(0).getType());
        const auto maxSlotsSum = VPUIP::getBarrierMaxVariantSum(nceOp);
        if (KX == 3 && SX == 1 && workloadChannelsMeetRequirement && workloadNumInTotal < maxSlotsSum &&
            !hasSparseInput) {
            supportedChannels = {VPU::NCEInvariant::VPU_CHANNEL_SIZE_FOR_L1OPT};
        }
    }

    _log.trace("getSupportedChannels: supportedChannels {0} on 40XX for nceOp {1}", supportedChannels,
               (*nceOps.begin())->getLoc());
    return supportedChannels;
}

namespace {

//
// CorrectNCEWorkloads
//

class CorrectNCEWorkloadsPass final : public VPU::arch40xx::impl::CorrectNCEWorkloadsBase<CorrectNCEWorkloadsPass> {
public:
    explicit CorrectNCEWorkloadsPass(Logger log): _log(log) {
        _log.setName(Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

private:
    Logger _log;
};

void CorrectNCEWorkloadsPass::safeRunOnFunc() {
    auto func = getOperation();
    WorkloadSplitter40XX splitter(func, _log);

    const auto arch = getArch(func);
    auto sparsityConstraint = VPU::getSparsityConstraint(arch);
    splitter.correctInvalidWorkload(sparsityConstraint);
}

}  // namespace

//
// createCorrectNCEWorkloadsPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::arch40xx::createCorrectNCEWorkloadsPass(Logger log) {
    return std::make_unique<CorrectNCEWorkloadsPass>(log);
}
