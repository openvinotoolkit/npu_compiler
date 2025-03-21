//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPU/utils/manual_strategy_utils.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include <mlir/IR/IRMapping.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_WRAPVERTICALFUSIONREGION
#define GEN_PASS_DEF_WRAPVERTICALFUSIONREGION
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;
using namespace VPU;

namespace {

void wrapIntoVFRegion(VPU::VerticalFusionOpInterface op, Logger log) {
    if (op->getParentOfType<VPU::VerticalFusionOp>() != nullptr) {
        log.trace("[SKIP] The Operation already wrapped into VF region");
        return;
    }

    const auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    const SmallVector<int64_t> one(inputType.getRank(), 1);

    auto tilingStrategyArray = op->hasAttr(tilingStrategy) ? op->getAttr(tilingStrategy).cast<mlir::ArrayAttr>()
                                                           : getIntArrayAttr(op->getContext(), one);

    const auto bodyBuilder = [op](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange newOperands) {
        mlir::IRMapping mapper;
        mapper.map(op->getOperands(), newOperands);
        auto* newOp = builder.clone(*op, mapper);
        newOp->removeAttr(tilingStrategy);
        builder.create<VPU::YieldOp>(loc, newOp->getResults());
    };

    OpBuilderLogger builderLog(log.nest());
    mlir::OpBuilder builder(op, &builderLog);

    auto vfOp = builder.create<VPU::VerticalFusionOp>(op->getLoc(), op->getResultTypes(), op->getOperands(),
                                                      bodyBuilder, tilingStrategyArray);
    op->replaceAllUsesWith(vfOp);
    op->erase();
}

//
// WrapVerticalFusionRegionPass
//

class WrapVerticalFusionRegionPass final :
        public VPU::impl::WrapVerticalFusionRegionBase<WrapVerticalFusionRegionPass> {
public:
    explicit WrapVerticalFusionRegionPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

//
// safeRunOnModule
//

void WrapVerticalFusionRegionPass::safeRunOnFunc() {
    const auto callback = [&](VPU::VerticalFusionOpInterface op) {
        if (mlir::isa<VPU::VerticalFusionOp>(op->getParentOp())) {
            _log.trace("Skip for operation '{0}' at '{1}' which is wrapped in other op", op->getName(), op->getLoc());
            return;
        }

        if (!op.isVFSupported()) {
            _log.trace("Skip for operation '{0}' at '{1}' which doesn't support VF", op->getName(), op->getLoc());
            return;
        }

        if (op->hasAttr(tilingStrategy)) {
            const auto tilingShape =
                    Shape(parseIntArrayAttr<int64_t>(op->getAttr(tilingStrategy).cast<mlir::ArrayAttr>()));
            auto tilingDimCount = getNonOneDim(tilingShape).size();
            if (tilingDimCount > 1) {
                _log.trace("Skip for operation '{0}' at '{1}' because VF doesn't support multi-dim tiling",
                           op->getName(), op->getLoc());
                return;
            }
        }

        _log.trace("Process Layer Operation '{0}' at '{1}'", op->getName(), op->getLoc());
        wrapIntoVFRegion(op, _log.nest());
    };

    getOperation().walk(callback);
}

}  // namespace

//
// createWrapVerticalFusionRegion
//

std::unique_ptr<mlir::Pass> VPU::createWrapVerticalFusionRegionPass(Logger log) {
    return std::make_unique<WrapVerticalFusionRegionPass>(log);
}
