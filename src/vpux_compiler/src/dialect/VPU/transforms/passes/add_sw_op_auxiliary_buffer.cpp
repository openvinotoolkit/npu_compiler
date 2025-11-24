//
// Copyright (C) 2024-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/dialect.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops_interfaces.hpp"
#include "vpux/compiler/dialect/VPU/transforms/passes.hpp"
#include "vpux/compiler/dialect/config/IR/resources.hpp"
#include "vpux/compiler/dialect/const/utils/utils.hpp"
#include "vpux/compiler/utils/rewriter.hpp"

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Pass/PassManager.h>

namespace vpux::VPU {
#define GEN_PASS_DECL_ADDSWOPAUXILIARYBUFFER
#define GEN_PASS_DEF_ADDSWOPAUXILIARYBUFFER
#include "vpux/compiler/dialect/VPU/passes.hpp.inc"
}  // namespace vpux::VPU

using namespace vpux;

namespace {

mlir::Value createAuxiliaryBuffer(mlir::Operation* op, mlir::Type type) {
    mlir::OpBuilder builder(op);
    const auto loc = appendLoc(op->getLoc(), "_aux");
    auto ndType = mlir::cast<NDTypeInterface>(type);
    if (mlir::isa<mlir::Float32Type>(ndType.getElementType())) {
        std::vector<float> vals(ndType.getShape().totalSize(), 0.0f);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    } else if (ndType.getElementType() == getUInt32Type(op->getContext())) {
        std::vector<uint32_t> vals(ndType.getShape().totalSize(), 0);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    } else {
        std::vector<uint8_t> vals(ndType.getTotalAllocSize().count(), 0);
        return Const::createConst(builder, loc, mlir::cast<mlir::RankedTensorType>(type), ArrayRef(vals));
    }
}

class AddSwOpAuxiliaryBufferPass final : public VPU::impl::AddSwOpAuxiliaryBufferBase<AddSwOpAuxiliaryBufferPass> {
public:
    explicit AddSwOpAuxiliaryBufferPass(const Logger& log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final {
        auto func = getOperation();

        auto result = func.walk([&](VPU::AuxiliaryBufferOpInterface op) {
            _log.trace("Got operation '{0}' at '{1}'", op->getName(), op->getLoc());
            const auto bufferTypes = op.getBufferTypes();
            if (bufferTypes.empty()) {
                _log.nest().trace("Missing auxiliary buffer types");
                return mlir::WalkResult::advance();
            }
            SmallVector<mlir::Value> buffers;
            for (auto bufferType : bufferTypes) {
                buffers.push_back(createAuxiliaryBuffer(op, bufferType));
            }
            if (mlir::failed(op.setAuxiliaryBuffers(buffers))) {
                _log.nest().debug("Failed to set auxiliary buffers: {0}", buffers);
                return mlir::WalkResult::interrupt();
            }
            _log.nest().trace("Added {0} auxiliary buffer(s)", buffers.size());
            return mlir::WalkResult::advance();
        });

        if (result.wasInterrupted()) {
            signalPassFailure();
            return;
        }
    }
};

}  // namespace

//
// createAddSwOpAuxiliaryBufferPass
//

std::unique_ptr<mlir::Pass> vpux::VPU::createAddSwOpAuxiliaryBufferPass(const Logger& log) {
    return std::make_unique<AddSwOpAuxiliaryBufferPass>(log);
}
