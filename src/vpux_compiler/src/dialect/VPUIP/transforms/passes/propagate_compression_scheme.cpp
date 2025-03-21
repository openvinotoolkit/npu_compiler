//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/swizzling_utils.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_PROPAGATESPARSITYCOMPRESSION
#define GEN_PASS_DEF_PROPAGATESPARSITYCOMPRESSION
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {

//
// PropagateSparsityCompression
//

class PropagateSparsityCompression final :
        public VPUIP::impl::PropagateSparsityCompressionBase<PropagateSparsityCompression> {
public:
    explicit PropagateSparsityCompression(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;

    void reinferOutputType(mlir::Operation* op);
    void reinferInnerBlockTypes(VPUIP::NCEClusterTilingOp clusterTilingOp,
                                VPUIP::SparsityCompressionAttr sparsityCompressionAttr);
    void propagateUpSparsityCompression(mlir::Value operand, VPUIP::SparsityCompressionAttr sparsityCompressionAttr);
    void propagateDownSparsityCompression(mlir::Operation* op, VPUIP::SparsityCompressionAttr sparsityCompressionAttr);
};

void PropagateSparsityCompression::reinferOutputType(mlir::Operation* op) {
    if (mlir::isa<mlir::InferTypeOpInterface>(op)) {
        vpux::inferReturnTypes(op, vpux::InferShapedTypeMode::ALL);
    } else if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
        for (auto p : op->getResults() | indexed) {
            auto resultIdx = p.index();
            auto result = p.value();
            auto outputOperand = VPUIP::getLayerViewSource(op, resultIdx);
            result.setType(outputOperand.getType());
        }
    }
}

// Reinfers the types inside the inner block of a cluster tiling operation so that the compact types
// contain the compression scheme of the outer operands
void PropagateSparsityCompression::reinferInnerBlockTypes(VPUIP::NCEClusterTilingOp clusterTilingOp,
                                                          VPUIP::SparsityCompressionAttr sparsityCompressionAttr) {
    // Find the compact types for the new arguments and their locations
    SmallVector<mlir::Type> newArgTypes;
    SmallVector<mlir::Location> newArgLocations;
    auto& block = clusterTilingOp.getBody().front();
    const auto operandTypes = clusterTilingOp.getOperandTypes();
    const auto blockArgs = block.getArguments();
    for (auto p : zip(operandTypes, blockArgs)) {
        const auto operandType = std::get<0>(p);
        const auto arg = std::get<1>(p);
        newArgLocations.push_back(arg.getLoc());

        mlir::Type newArgType = operandType;
        if (auto distType = operandType.dyn_cast<VPUIP::DistributedBufferType>()) {
            newArgType = distType.getCompactType();
        } else if (auto sparseType = operandType.dyn_cast<VPUIP::SparseBufferType>()) {
            if (auto distDataType = sparseType.getData().dyn_cast<VPUIP::DistributedBufferType>()) {
                mlir::MemRefType dataType = distDataType.getCompactType();
                mlir::MemRefType smType = nullptr;
                if (sparseType.getSparsityMap() != nullptr &&
                    sparseType.getSparsityMap().isa<VPUIP::DistributedBufferType>()) {
                    smType = sparseType.getSparsityMap().cast<VPUIP::DistributedBufferType>().getCompactType();
                }
                mlir::MemRefType seType = nullptr;
                if (sparseType.getStorageElementTable() != nullptr &&
                    sparseType.getStorageElementTable().isa<VPUIP::DistributedBufferType>()) {
                    seType = sparseType.getStorageElementTable().cast<VPUIP::DistributedBufferType>().getCompactType();
                }
                newArgType = VPUIP::SparseBufferType::get(dataType, smType, seType, sparseType.getIsWeights(),
                                                          sparseType.getSparsityCompression());
            }
        }
        newArgTypes.push_back(newArgType);
    }

    auto origArgCount = block.getArguments().size();

    // Add the new arguments and replace the uses of the original ones
    for (auto p : zip(newArgTypes, newArgLocations) | indexed) {
        auto type = std::get<0>(p.value());
        auto loc = std::get<1>(p.value());
        auto newArg = block.addArgument(type, loc);
        block.getArgument(checked_cast<unsigned int>(p.index())).replaceAllUsesWith(newArg);
    }

    // Erase the original arguments
    while (origArgCount > 0) {
        block.eraseArgument(0);
        origArgCount--;
    }

    // Propagate the compression scheme inside the block of the cluster tiling operation
    auto firstOp = &block.front();
    propagateDownSparsityCompression(firstOp, sparsityCompressionAttr);
}

// Propagates the compression scheme attribute upwards, until an operation without operands is reached (e.g. allocation)
void PropagateSparsityCompression::propagateUpSparsityCompression(
        mlir::Value operand, VPUIP::SparsityCompressionAttr sparsityCompressionAttr) {
    auto parentOp = operand.getDefiningOp();
    if (parentOp == nullptr || parentOp->getNumOperands() == 0) {
        auto newType = VPUIP::setSparsityCompressionAttr(operand.getType(), sparsityCompressionAttr);
        operand.setType(newType);
        return;
    }

    if (mlir::isa<vpux::GroupedViewOpInterface>(parentOp)) {
        propagateUpSparsityCompression(parentOp->getOperand(0), sparsityCompressionAttr);
    } else {
        for (auto operand : parentOp->getOperands()) {
            propagateUpSparsityCompression(operand, sparsityCompressionAttr);
        }
    }

    reinferOutputType(parentOp);
}

// Propagates the compression scheme attribute to all user operations, until either an NCE operation is reached or the
// end of the model
void PropagateSparsityCompression::propagateDownSparsityCompression(
        mlir::Operation* op, VPUIP::SparsityCompressionAttr sparsityCompressionAttr) {
    if (mlir::isa<VPUIP::NCEClusterTaskOp, mlir::func::ReturnOp>(op)) {
        return;
    }

    auto clusterTilingOp = mlir::dyn_cast<VPUIP::NCEClusterTilingOp>(op);
    if (clusterTilingOp != nullptr && mlir::isa<VPUIP::NCEClusterTaskOp>(clusterTilingOp.getInnerTaskOp())) {
        reinferInnerBlockTypes(clusterTilingOp, sparsityCompressionAttr);
        return;
    }

    if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
        for (auto resultIdx : irange(op->getResults().size())) {
            auto outputOperand = VPUIP::getLayerViewSource(op, resultIdx);
            propagateUpSparsityCompression(outputOperand, sparsityCompressionAttr);
        }
    }

    if (clusterTilingOp != nullptr) {
        reinferInnerBlockTypes(clusterTilingOp, sparsityCompressionAttr);
    }

    reinferOutputType(op);

    for (auto userOp : op->getUsers()) {
        propagateDownSparsityCompression(userOp, sparsityCompressionAttr);
    }
}

void PropagateSparsityCompression::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](Const::DeclareOp constOp) {
        const auto& contentAttr = constOp.getContentAttr();
        const auto transformations = contentAttr.getTransformations();
        if (transformations.empty()) {
            return;
        }

        auto sparsifyTransformationIt =
                std::find_if(transformations.rbegin(), transformations.rend(), [](Const::TransformAttrInterface tr) {
                    return tr.isa<Const::SparsifyAttr>();
                });
        if (sparsifyTransformationIt == transformations.rend()) {
            return;
        }

        auto userOp = *constOp.getOutput().getUsers().begin();
        auto userGroupOp = mlir::dyn_cast<VPUIP::GroupSparseBufferOp>(userOp);
        VPUX_THROW_UNLESS(userGroupOp != nullptr, "Expected weights user to be a VPUIP.GroupSparseBuffer op, got {0}",
                          userOp);
        auto sparsityCompressionAttr = userGroupOp.getSparsityCompressionAttr();

        const auto outputType = constOp.getType().cast<vpux::NDTypeInterface>();
        const auto newOutputType = getMemRefType(
                outputType.getShape(), outputType.getElementType(), outputType.getDimsOrder(), outputType.getMemSpace(),
                outputType.getStrides(), vpux::getSwizzlingSchemeAttr(outputType), sparsityCompressionAttr);

        constOp.getOutput().setType(newOutputType);

        for (auto userOp : constOp.getOutput().getUsers()) {
            auto groupOp = mlir::dyn_cast<VPUIP::GroupSparseBufferOp>(userOp);
            VPUX_THROW_UNLESS(groupOp != nullptr, "Expected weights user to be a VPUIP.GroupSparseBuffer op, got {0}",
                              userOp);
            VPUX_THROW_UNLESS(sparsityCompressionAttr == groupOp.getSparsityCompressionAttr(),
                              "Mismatch between the compression scheme of constant op '{0}' and grouping op '{1}'",
                              sparsityCompressionAttr, groupOp.getSparsityCompressionAttr());
            propagateDownSparsityCompression(userOp, sparsityCompressionAttr);
        }
    });
}

}  // namespace

//
// createPropagateSparsityCompressionPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createPropagateSparsityCompressionPass(Logger log) {
    return std::make_unique<PropagateSparsityCompression>(log);
}
