//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/Shave/IR/ops/meta-ops.hpp"

using namespace vpux;

//
// KernelRegionOp
//

mlir::LogicalResult vpux::Shave::KernelRegionOp::verify() {
    assert(getBody().hasOneBlock() && "SingleBlock trait guarantees exactly one block");
    auto& entryBlock = getBody().front();
    const auto inputs = getInputs();
    const auto blockArgs = entryBlock.getArguments();

    if (blockArgs.size() != inputs.size()) {
        return emitOpError() << "entry block has " << blockArgs.size() << " argument(s) but op has " << inputs.size()
                             << " input(s)";
    }

    for (auto [idx, inputVal, blockArg] : llvm::enumerate(inputs, blockArgs)) {
        if (inputVal.getType() != blockArg.getType()) {
            return emitOpError() << "type mismatch at input " << idx << ": op input type " << inputVal.getType()
                                 << " does not match block argument type " << blockArg.getType();
        }
    }

    auto* terminator = entryBlock.getTerminator();
    if (terminator == nullptr) {
        return emitOpError() << "body entry block has no terminator";
    }

    auto yieldOp = mlir::dyn_cast<Shave::KernelRegionYieldOp>(terminator);
    if (yieldOp == nullptr) {
        return emitOpError() << "body must be terminated by a Shave.KernelRegion.Yield op, got "
                             << terminator->getName();
    }

    const auto resultTypes = getResultTypes();
    const auto yieldedValues = yieldOp.getValues();

    if (yieldedValues.size() != resultTypes.size()) {
        return emitOpError() << "yields " << yieldedValues.size() << " value(s) but op has " << resultTypes.size()
                             << " result(s)";
    }

    for (auto [idx, yieldedVal, resultType] : llvm::enumerate(yieldedValues, resultTypes)) {
        if (yieldedVal.getType() != resultType) {
            return emitOpError() << "type mismatch at result " << idx << ": yielded type " << yieldedVal.getType()
                                 << " does not match op result type " << resultType;
        }
    }

    return mlir::success();
}
