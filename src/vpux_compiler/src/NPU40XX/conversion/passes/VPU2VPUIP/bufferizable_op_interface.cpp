//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/NPU40XX/conversion/passes/VPU2VPUIP/bufferizable_op_interface.hpp"
#include "vpux/compiler/NPU40XX/utils.hpp"
#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferize_sw_ops_interface.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"

#include <mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h>
#include <mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>

using namespace vpux;

namespace {

class GatherDMAOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<GatherDMAOpBufferizeModel, VPU::GatherDMAOp> {
public:
    mlir::LogicalResult bufferizeImpl(VPU::GatherDMAOp origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions& options,
                                      VPU::GatherDMAOp::Adaptor adaptor) const;
};

mlir::LogicalResult GatherDMAOpBufferizeModel::bufferizeImpl(VPU::GatherDMAOp origOp, mlir::RewriterBase& rewriter,
                                                             const mlir::bufferization::BufferizationOptions&,
                                                             VPU::GatherDMAOp::Adaptor adaptor) const {
    return vpux::bufferizeOp(origOp->getContext(), origOp, adaptor, rewriter);
}

void registerGatherDMAOpBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, VPU::VPUDialect*, VPUIP::VPUIPDialect*) {
        VPU::GatherDMAOp::attachInterface<GatherDMAOpBufferizeModel>(*ctx);
    });
}

}  // namespace

//
// registerBufferizableOpInterfaces
//

void vpux::arch40xx::registerBufferizableOpInterfaces(mlir::DialectRegistry& registry) {
    vpux::registerConstDeclareBufferizableOpInterfaces(registry);
    vpux::registerCoreBufferizableOpInterfaces(registry);
    vpux::registerFuncAndReturnBufferizableOpInterfaces(registry);
    vpux::registerSoftwareLayerBufferizableOpInterfaces(registry);
    vpux::registerVpuNceBufferizableOpInterfaces(registry);
    vpux::registerVPUBufferizableOpInterfaces(registry);
    registerGatherDMAOpBufferizableOpInterfaces(registry);

    mlir::memref::registerAllocationOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
}
