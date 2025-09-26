//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU37XX/dialect/VPUIP/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/interfaces/workload_splitter.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/sparsity_constraint.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

namespace {

template <class MainOpType>
class AlignedWorkloadChannelsOpModel final :
        public VPU::AlignedWorkloadChannelsOpInterface::ExternalModel<AlignedWorkloadChannelsOpModel<MainOpType>,
                                                                      MainOpType> {
public:
    SmallVector<int64_t> getSupportedWorkLoadChannels(mlir::Operation* nceOp) const {
        auto func = nceOp->getParentOfType<mlir::func::FuncOp>();
        auto log = Logger::global();
        const auto arch = config::getArch(func);
        auto sparsityConstraint = VPU::getSparsityConstraint(arch);
        VPU::WorkloadSplitter splitter(func, vpux::VPU::supportedChannelsDW, log);

        // More than one operation might need to be handled at the same time for some sparse activations,
        // to satisfy the requirements of the consumer ops
        mlir::DenseSet<mlir::Operation*> producerNCEOps{nceOp};
        const auto invalidSparseOps =
                splitter.findInvalidSparseOps(mlir::cast<VPU::NCEOpInterface>(nceOp), sparsityConstraint);
        if (!invalidSparseOps.empty()) {
            producerNCEOps.clear();
            producerNCEOps.insert(invalidSparseOps.begin(), invalidSparseOps.end());
        }

        const auto supportedChannels = splitter.getSupportedChannels(producerNCEOps, sparsityConstraint);
        log.trace("getSupportedWorkLoadChannels: supportedChannels {0} on 37XX for nceOp '{1}'", supportedChannels,
                  nceOp->getLoc());
        return supportedChannels;
    }
};
}  // namespace

void vpux::VPUIP::arch37xx::registerAlignedWorkloadChannelsOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*) {
        VPU::NCEDepthConvolutionOp::attachInterface<AlignedWorkloadChannelsOpModel<VPU::NCEDepthConvolutionOp>>(*ctx);
        VPU::NCEMaxPoolOp::attachInterface<AlignedWorkloadChannelsOpModel<VPU::NCEMaxPoolOp>>(*ctx);
        VPU::NCEAveragePoolOp::attachInterface<AlignedWorkloadChannelsOpModel<VPU::NCEAveragePoolOp>>(*ctx);
    });
}
