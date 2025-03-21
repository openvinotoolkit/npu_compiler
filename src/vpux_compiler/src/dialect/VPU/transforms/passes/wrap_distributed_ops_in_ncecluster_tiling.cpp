//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/IE/utils/dynamic_shape_utils.hpp"
#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/distributed_tensor_utils.hpp"

#include <mlir/IR/IRMapping.h>
#include <mlir/Transforms/DialectConversion.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_WRAPDISTRIBUTEDOPSINNCECLUSTERTILING
#define GEN_PASS_DEF_WRAPDISTRIBUTEDOPSINNCECLUSTERTILING
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

//
// WrapDistributedOpsInNCEClusterTiling
//

class WrapDistributedOpsInNCEClusterTiling final :
        public VPU::impl::WrapDistributedOpsInNCEClusterTilingBase<WrapDistributedOpsInNCEClusterTiling> {
public:
    explicit WrapDistributedOpsInNCEClusterTiling(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//
void WrapDistributedOpsInNCEClusterTiling::safeRunOnFunc() {
    auto func = getOperation();

    func->walk([&](mlir::Operation* origOp) {
        if (!mlir::isa<ClusteredOpInterface, VPU::CopyOp>(origOp)) {
            return mlir::WalkResult::skip();
        }

        if (mlir::isa<VPU::ConcatOp>(origOp)) {
            return mlir::WalkResult::skip();
        }

        if (origOp->getParentOfType<NCEClusterTilingOp>() != nullptr) {
            _log.trace("Op {0} already wrapped into NCEClusterTilingOp", origOp->getName());
            return mlir::WalkResult::skip();
        }

        // Check for any distributed operands
        auto hasDistributedOperand = llvm::any_of(origOp->getOperands().getTypes(), [](mlir::Type t) {
            if (auto checkDistributed = t.dyn_cast<VPU::DistributedTypeInterface>()) {
                return checkDistributed.containsDistributedTypes();
            }
            return false;
        });
        // Check for any distributed results
        bool hasDistributedResult = llvm::any_of(origOp->getResults().getTypes(), [](mlir::Type t) {
            if (auto checkDistributed = t.dyn_cast<VPU::DistributedTypeInterface>()) {
                return checkDistributed.containsDistributedTypes();
            }
            return false;
        });
        // If the op doesn't have DistributedTensor I/O, continue
        if (!hasDistributedOperand && !hasDistributedResult) {
            return mlir::WalkResult::skip();
        }

        if (IE::hasDynamicTensors(origOp)) {
            return mlir::WalkResult::skip();
        }

        mlir::OpBuilder nceBuilder(origOp);
        const auto bodyBuilder = [origOp](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
            mlir::IRMapping mapper;
            mapper.map(origOp->getOperands(), newOperands);
            auto* newOp = builder.clone(*origOp, mapper);
            for (auto operandIter : newOperands | indexed) {
                newOp->setOperand(operandIter.index(), newOperands[operandIter.index()]);
            }

            for (auto resultIter : newOp->getResults()) {
                resultIter.setType(vpux::VPU::getCompactTypeFromDistributed(resultIter.getType()));
            }
            builder.create<YieldOp>(loc, newOp->getResults());
        };

        _log.trace("Wrap {0} into NCEClusterTilingOp", origOp->getName());

        auto origOpOperands = origOp->getOperands();
        if (mlir::isa<VPU::NCEEltwiseOp>(origOp)) {
            // Special case for Eltwise with identical input operands
            if (origOp->getOperand(0) == origOp->getOperand(1)) {
                origOpOperands = origOpOperands.take_front(1);
            }
        }
        nceBuilder.setInsertionPoint(origOp);
        auto nceClusterOp = nceBuilder.create<vpux::VPU::NCEClusterTilingOp>(origOp->getLoc(), origOp->getResultTypes(),
                                                                             origOpOperands, bodyBuilder);

        origOp->replaceAllUsesWith(nceClusterOp.getResults());
        origOp->erase();

        return mlir::WalkResult::advance();
    });
}

}  // namespace

//
// createWrapDistributedOpsInNCEClusterTiling
//

std::unique_ptr<mlir::Pass> VPU::createWrapDistributedOpsInNCEClusterTiling(Logger log) {
    return std::make_unique<WrapDistributedOpsInNCEClusterTiling>(log);
}
