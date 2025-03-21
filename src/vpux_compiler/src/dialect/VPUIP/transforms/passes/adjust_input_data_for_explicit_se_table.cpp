//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_ADJUSTINPUTDATAFOREXPLICITSETABLE
#define GEN_PASS_DEF_ADJUSTINPUTDATAFOREXPLICITSETABLE
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// AdjustInputDataForExplicitSETablePass
//

class AdjustInputDataForExplicitSETablePass final :
        public VPUIP::impl::AdjustInputDataForExplicitSETableBase<AdjustInputDataForExplicitSETablePass> {
public:
    explicit AdjustInputDataForExplicitSETablePass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
};

VPU::DistributionInfoAttr updateExplicitDistributedAttrWithSpecificDim(VPU::DistributionInfoAttr dataDistributedAttr,
                                                                       VPU::DistributionInfoAttr seDistributedAttr,
                                                                       Dim dim, mlir::MLIRContext* ctx) {
    auto updateShapes = [&](mlir::ArrayAttr origShapesAttr, mlir::ArrayAttr targetShapesAttr) {
        auto origShapes = parseIntArrayOfArrayAttr<int64_t>(origShapesAttr);
        auto targetShapes = parseIntArrayOfArrayAttr<int64_t>(targetShapesAttr);
        SmallVector<SmallVector<int64_t>> newShapes;
        newShapes.reserve(origShapes.size());
        for (auto clusterId : irange(origShapes.size())) {
            auto newShape = targetShapes[clusterId];
            newShape[dim.ind()] = origShapes[clusterId][dim.ind()];
            newShapes.push_back(newShape);
        }
        return newShapes;
    };

    const auto newComputeShapes =
            updateShapes(dataDistributedAttr.getComputeShapes(), seDistributedAttr.getComputeShapes());
    const auto newComputeOffsets =
            updateShapes(dataDistributedAttr.getComputeOffsets(), seDistributedAttr.getComputeOffsets());
    const auto newMemoryShapes =
            updateShapes(dataDistributedAttr.getMemoryShapes(), seDistributedAttr.getMemoryShapes());
    const auto newMemoryOffsets =
            updateShapes(dataDistributedAttr.getMemoryOffsets(), seDistributedAttr.getMemoryOffsets());

    return VPU::DistributionInfoAttr::get(
            ctx, dataDistributedAttr.getMode(), dataDistributedAttr.getNumTiles(), nullptr, nullptr, nullptr,
            dataDistributedAttr.getNumClusters(), dataDistributedAttr.getAlignment(),
            dataDistributedAttr.getUniformDistributedSegments(), getIntArrayOfArray(ctx, newComputeShapes),
            getIntArrayOfArray(ctx, newComputeOffsets), getIntArrayOfArray(ctx, newMemoryShapes),
            getIntArrayOfArray(ctx, newMemoryOffsets), dataDistributedAttr.getEqualMemoryAndComputeView());
}

void AdjustInputDataForExplicitSETablePass::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](VPUIP::NCEClusterTaskOp nceOp) {
        if (nceOp.getInputStorageElementTable() == nullptr) {
            return;
        }

        _log.trace("Got '{0}' at '{1}'", nceOp->getName(), nceOp->getLoc());
        VPUX_THROW_UNLESS(nceOp.getInputSeSize().has_value(), "Missing input storage element size");

        auto getNewType = [&](VPURT::DeclareBufferOp declareOp, mlir::Value seOperand) {
            auto seOperandType = seOperand.getType().cast<vpux::NDTypeInterface>();
            auto dataType = declareOp.getType().cast<vpux::NDTypeInterface>();
            auto newShape = Shape(seOperandType.getShape().raw());
            newShape[Dims4D::Act::C] *= nceOp.getInputSeSize().value();

            auto dataDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(dataType);
            auto seDistributedType = mlir::dyn_cast<VPUIP::DistributedBufferType>(seOperandType);
            if (dataDistributedType != nullptr && seDistributedType != nullptr) {
                auto dataDistributedAttr = dataDistributedType.getDistribution();
                auto seDistributedAttr = seDistributedType.getDistribution();
                if (dataDistributedAttr.getMemoryShapes() != nullptr &&
                    seDistributedAttr.getMemoryShapes() != nullptr) {
                    auto newDistributedAttr = updateExplicitDistributedAttrWithSpecificDim(
                            dataDistributedAttr, seDistributedAttr, Dims4D::Act::C, nceOp.getContext());
                    return dataDistributedType.changeShapeForExplicitDistribution(newShape, newDistributedAttr);
                }
            }

            return dataType.changeShape(newShape);
        };

        auto adaptTypeFor = [&](mlir::Value operand, mlir::Value seOperand) {
            auto declareOp = operand.getDefiningOp<VPURT::DeclareBufferOp>();
            VPUX_THROW_UNLESS(declareOp != nullptr, "Expected buffer declaration, got {0}", operand.getDefiningOp());
            mlir::OpBuilder builder(declareOp);

            auto newDeclareOp = builder.clone(*declareOp.getOperation());
            auto newType = getNewType(declareOp, seOperand);
            newDeclareOp->getResult(0).setType(newType);

            declareOp.getBuffer().replaceUsesWithIf(newDeclareOp->getResult(0), [&](mlir::OpOperand& operand) -> bool {
                return operand.getOwner() == nceOp.getOperation();
            });
            if (declareOp.getBuffer().use_empty()) {
                declareOp->erase();
            }
        };

        adaptTypeFor(nceOp.getInput(), nceOp.getInputStorageElementTable());
        adaptTypeFor(nceOp.getParentInput(), nceOp.getParentInputStorageElementTable());
    });
}

}  // namespace

//
// createAdjustInputDataForExplicitSETablePass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createAdjustInputDataForExplicitSETablePass(Logger log) {
    return std::make_unique<AdjustInputDataForExplicitSETablePass>(log);
}
