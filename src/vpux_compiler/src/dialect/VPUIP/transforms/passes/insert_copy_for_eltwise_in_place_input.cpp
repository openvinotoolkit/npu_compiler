//
// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/core/aliases_info.hpp"
#include "vpux/compiler/core/attributes/stride_reqs.hpp"
#include "vpux/compiler/dialect/VPUIP/transforms/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/utils.hpp"
#include "vpux/compiler/dialect/core/interfaces/type_interfaces.hpp"
#include "vpux/compiler/utils/rewriter.hpp"
#include "vpux/compiler/utils/types.hpp"

namespace vpux::VPUIP {
#define GEN_PASS_DECL_INSERTCOPYFORELTWISEINPLACEINPUT
#define GEN_PASS_DEF_INSERTCOPYFORELTWISEINPLACEINPUT
#include "vpux/compiler/dialect/VPUIP/passes.hpp.inc"
}  // namespace vpux::VPUIP

using namespace vpux;

namespace {
vpux::Strides getCompactStrides(NDTypeInterface type) {
    const Bit elemTypeSize = vpux::getElemTypeSize(type.getElementType());
    const auto dimsOrder = type.getDimsOrder();
    const auto memShape = dimsOrder.toMemoryOrder(Shape(type.getShape()));
    const auto memStrides = StrideReqs::compact(dimsOrder.numDims()).calcStrides(elemTypeSize, memShape);
    return dimsOrder.toLogicalOrder(memStrides);
}

//
// InsertCopyForEltwiseInPlaceInputPass
//

class InsertCopyForEltwiseInPlaceInputPass final :
        public VPUIP::impl::InsertCopyForEltwiseInPlaceInputBase<InsertCopyForEltwiseInPlaceInputPass> {
public:
    explicit InsertCopyForEltwiseInPlaceInputPass(Logger log) {
        Base::initLogger(log, Base::getArgumentName());
    }

private:
    void safeRunOnFunc() final;
    std::optional<mlir::Value> getInputToOverwrite(VPUIP::NCEClusterTaskOp op, AliasesInfo& aliasesInfo);
    void insertCopies(VPUIP::NCEClusterTaskOp eltwise, mlir::Value overwrittenInput);
};

std::optional<mlir::Value> InsertCopyForEltwiseInPlaceInputPass::getInputToOverwrite(VPUIP::NCEClusterTaskOp op,
                                                                                     AliasesInfo& aliasesInfo) {
    if (op.getTaskType() != VPUIP::NCETaskType::ELTWISE) {
        return std::nullopt;
    }

    if (!op.getIsInplace().value_or(false)) {
        return std::nullopt;
    }

    _log.trace("Found in place Eltwise {0}", op.getLoc());

    auto eltwiseInputs = VPUIP::getLayerInputs(op);
    auto eltwiseOutput = VPUIP::getLayerOutputs(op)[0];

    auto outputRootBuff = aliasesInfo.getRoot(eltwiseOutput);

    auto overwrittenInput = llvm::find_if(eltwiseInputs, [&](mlir::Value input) -> bool {
        auto inputRootBuff = aliasesInfo.getRoot(input);

        return inputRootBuff == outputRootBuff;
    });

    VPUX_THROW_WHEN(overwrittenInput == eltwiseInputs.end(), "In place eltwise has no overwritten input.");

    return *overwrittenInput;
}

void InsertCopyForEltwiseInPlaceInputPass::insertCopies(VPUIP::NCEClusterTaskOp eltwise, mlir::Value overwrittenInput) {
    mlir::OpBuilder builder(eltwise);
    auto inDistributedType = overwrittenInput.getType().dyn_cast<VPUIP::DistributedBufferType>();
    NDTypeInterface inputType = nullptr;
    if (inDistributedType != nullptr) {
        // DistributedBuffer
        inputType = mlir::cast<vpux::NDTypeInterface>(inDistributedType.getCompactType());
    } else {
        // memref
        inputType = mlir::cast<vpux::NDTypeInterface>(overwrittenInput.getType());
    }

    // To DDR
    const auto compactStrides = getCompactStrides(inputType);
    auto newDDRType = inputType.changeMemSpace(VPU::MemoryKind::DDR).changeStrides(compactStrides);
    auto newAllocDDROp = builder.create<mlir::memref::AllocOp>(appendLoc(eltwise->getLoc(), "_elt_in_place_input_DDR"),
                                                               newDDRType.cast<mlir::MemRefType>());
    auto newCopyToDDR = builder.create<VPUIP::CopyOp>(appendLoc(eltwise->getLoc(), "_unique_consumer_spill"),
                                                      overwrittenInput, newAllocDDROp);

    // To CMX
    mlir::Value bufferResult = nullptr;
    if (inDistributedType != nullptr) {
        // DistributedBuffer
        auto newDistributedBuff = builder.create<VPURT::AllocDistributed>(
                appendLoc(eltwise->getLoc(), "_elt_in_place_input_CMX"), inDistributedType, nullptr, nullptr);
        bufferResult = newDistributedBuff->getResult(0);
    } else {
        // memref
        auto newAllocCMXOp = builder.create<mlir::memref::AllocOp>(
                appendLoc(eltwise->getLoc(), "_elt_in_place_input_CMX"), inputType.cast<mlir::MemRefType>());
        bufferResult = newAllocCMXOp->getResult(0);
    }
    auto newCopyToCMX = builder.create<VPUIP::CopyOp>(appendLoc(eltwise->getLoc(), "_unique_consumer_spill"),
                                                      newCopyToDDR.getResult(), static_cast<mlir::Value>(bufferResult));

    overwrittenInput.replaceUsesWithIf(newCopyToCMX->getResult(0), [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == eltwise;
    });

    auto eltwiseOutput = VPUIP::getLayerOutputs(eltwise)[0];
    // Ensure eltwise output element type is compatible
    auto outputType = mlir::cast<vpux::NDTypeInterface>(eltwiseOutput.getType());
    auto bufferResultType = mlir::cast<vpux::NDTypeInterface>(bufferResult.getType());
    auto isQuantElemType = mlir::dyn_cast<mlir::quant::QuantizedType>(outputType.getElementType());
    if (isQuantElemType && outputType.getElementType() != bufferResultType.getElementType()) {
        auto quantizeCastOp = builder.create<VPUIP::QuantizeCastOp>(appendLoc(eltwise->getLoc(), "_cast_output"),
                                                                    outputType, bufferResult);
        bufferResult = quantizeCastOp.getResult();
    }
    eltwiseOutput.replaceUsesWithIf(bufferResult, [&](mlir::OpOperand& opOperand) {
        return opOperand.getOwner() == eltwise;
    });
}

void InsertCopyForEltwiseInPlaceInputPass::safeRunOnFunc() {
    auto& aliasesInfo = getAnalysis<AliasesInfo>();
    auto func = getOperation();

    func->walk([&](VPUIP::NCEClusterTaskOp op) {
        auto possibleInputToOverwrite = getInputToOverwrite(op, aliasesInfo);
        if (!possibleInputToOverwrite.has_value()) {
            return;
        }

        auto nestedLog = _log.nest();
        auto overwrittenInput = possibleInputToOverwrite.value();

        // Check that there are no other consumers up until the first producer compute op (NCEClusterTask/SWOp) or the
        // first Copy op
        if (VPUIP::isEltwiseTheOnlyConsumer(op, overwrittenInput, /*checkThroughCopyOps=*/false, nestedLog)) {
            nestedLog.trace("Input of in place Eltwise can be safely overwritten by the output. in = {0}",
                            overwrittenInput);
            return;
        }

        nestedLog.trace("Input buffer of in place Eltwise {0} has another consumer that might get overwritten by "
                        "Eltwise's output.",
                        op.getLoc());
        insertCopies(op, overwrittenInput);

        nestedLog.trace("Inserted spilling copies.");
    });
}

}  // namespace

//
// createInsertCopyForEltwiseInPlaceInputPass
//

std::unique_ptr<mlir::Pass> vpux::VPUIP::createInsertCopyForEltwiseInPlaceInputPass(Logger log) {
    return std::make_unique<InsertCopyForEltwiseInPlaceInputPass>(log);
}
