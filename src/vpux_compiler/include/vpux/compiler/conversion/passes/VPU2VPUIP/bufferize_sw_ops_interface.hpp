//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/passes/VPU2VPUIP/bufferizable_ops_interface.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops_interfaces.hpp"

namespace vpux::VPU {
class Concat;
class PermuteCastOp;
class StridedSliceOp;
}  // namespace vpux::VPU

namespace vpux {

mlir::LogicalResult bufferizeSWLayerOp(mlir::RewriterBase& rewriter, mlir::ModuleOp module, mlir::Operation* op,
                                       ArrayRef<mlir::Value> newOperands, Logger log);
mlir::LogicalResult bufferizeDistributedSWLayerOp(mlir::RewriterBase& rewriter, mlir::ModuleOp module,
                                                  mlir::Operation* op, ArrayRef<mlir::Value> newOperands, Logger log);

bool canBeBufferizedToCopies(VPU::ConcatOp concatOp);
bool canBeBufferizedToCopies(VPU::StridedSliceOp stridedSliceOp);
bool canBeBufferizedToCast(VPU::PermuteCastOp op);

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

        const SmallVector<mlir::Value> bufferizedOperands(adaptor.getOperands().begin(), adaptor.getOperands().end());

        auto hasDistributedType = [](mlir::Value value) {
            if (auto distributedIf = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(value.getType())) {
                return distributedIf.containsDistributedTypes();
            }
            return false;
        };
        if (llvm::any_of(origOp->getOperands(), hasDistributedType) ||
            llvm::any_of(origOp->getResults(), hasDistributedType)) {
            return bufferizeDistributedSWLayerOp(rewriter, module, origOp, bufferizedOperands, log);
        } else {
            return bufferizeSWLayerOp(rewriter, module, origOp, bufferizedOperands, log);
        }
    }
};

}  // namespace vpux
