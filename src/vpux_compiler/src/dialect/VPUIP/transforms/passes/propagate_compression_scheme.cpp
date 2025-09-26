//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPUIP/IR/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPUIP/IR/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
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

    if (mlir::isa<VPUIP::LayerOpInterface>(op)) {
        for (auto resultIdx : irange(op->getResults().size())) {
            auto outputOperand = VPUIP::getLayerViewSource(op, resultIdx);
            propagateUpSparsityCompression(outputOperand, sparsityCompressionAttr);
        }
    }

    reinferOutputType(op);

    for (auto userOp : op->getUsers()) {
        propagateDownSparsityCompression(userOp, sparsityCompressionAttr);
    }
}

void PropagateSparsityCompression::safeRunOnFunc() {
    auto func = getOperation();

    func.walk([&](Const::DeclareOp constOp) {
        if (!Const::hasSparsifyTransformation(constOp)) {
            return;
        }

        auto userOp = *constOp.getOutput().getUsers().begin();
        auto userGroupOp = mlir::dyn_cast<VPUIP::GroupSparseBufferOp>(userOp);
        VPUX_THROW_UNLESS(userGroupOp != nullptr, "Expected weights user to be a VPUIP.GroupSparseBuffer op, got {0}",
                          userOp->getName());
        auto sparsityCompressionAttr = userGroupOp.getSparsityCompressionAttr();

        const auto outputType = mlir::cast<vpux::NDTypeInterface>(constOp.getType());
        const auto strides = outputType.getStrides();
        const auto newOutputType = getMemRefType(outputType.getShape(), outputType.getElementType(),
                                                 outputType.getDimsOrder(), outputType.getMemSpace(), strides,
                                                 vpux::getSwizzlingSchemeAttr(outputType), sparsityCompressionAttr);

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
