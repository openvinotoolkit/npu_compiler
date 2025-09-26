//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

namespace vpux {

mlir::LogicalResult bufferizeSWLayerOp(mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op,
                                       ArrayRef<mlir::Value> newOperands, Logger log);
mlir::LogicalResult bufferizeDistributedSWLayerOp(mlir::RewriterBase& rewriter, mlir::ModuleOp module,
                                                  mlir::Operation* op, ArrayRef<mlir::Value> newOperands, Logger log);

//
// SoftwareLayerOpBufferizeModel
//

// Common Software Layer Operation bufferize model, used by arch37xx+
template <typename MainOpType>
class SoftwareLayerOpBufferizeModel :
        public BufferizableOpInterfaceExternalModelBase<SoftwareLayerOpBufferizeModel<MainOpType>, MainOpType> {
public:
    mlir::LogicalResult bufferizeImpl(MainOpType origOp, mlir::RewriterBase& rewriter,
                                      const mlir::bufferization::BufferizationOptions&,
                                      typename MainOpType::Adaptor& adaptor) const {
        auto log = Logger::global().nest("one-shot-bufferize-SoftwareLayerOp", 0);
        log.trace("Got {0} at {1}", origOp->getName(), origOp->getLoc());

        constexpr bool opIsSwLayerOperation = MainOpType::template hasTrait<VPU::LayerOpInterface::Trait>() ||
                                              MainOpType::template hasTrait<VPUIP::SoftwareLayerOpInterface::Trait>();
        static_assert(opIsSwLayerOperation, "MainOpType is not a Software layer operation");

        auto module = origOp->template getParentOfType<mlir::ModuleOp>();
        if (module == nullptr) {
            return errorAt(origOp->getLoc(), "Operation {0} has no parent Module Op", origOp->getName());
        }

        auto valueRange = adaptor.getOperands();
        auto valueRangeBegin = valueRange.begin().getBase().template get<const mlir::Value*>();
        auto bufferizedOperands = mlir::ArrayRef<mlir::Value>(valueRangeBegin, valueRange.size());

        SmallVector<mlir::Type> mergedOperands;
        mergedOperands.reserve(origOp->getOperands().size() + origOp->getResults().size());
        mergedOperands.insert(mergedOperands.end(), origOp->getOperands().getTypes().begin(),
                              origOp->getOperands().getTypes().end());
        mergedOperands.insert(mergedOperands.end(), origOp->getResults().getTypes().begin(),
                              origOp->getResults().getTypes().end());
        auto hasDistributedOperand = llvm::any_of(mergedOperands, [](mlir::Type t) {
            if (auto checkDistributed = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(t)) {
                return checkDistributed.containsDistributedTypes();
            }
            return false;
        });
        if (!hasDistributedOperand) {
            return bufferizeSWLayerOp(rewriter, module, origOp, bufferizedOperands, log);
        } else {
            return bufferizeDistributedSWLayerOp(rewriter, module, origOp, bufferizedOperands, log);
        }
    }
};

}  // namespace vpux
