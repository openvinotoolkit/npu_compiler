//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelParamsRewriter::symbolize(VPUMI40XX::KernelParamsOp op, SymbolMapper&,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto context = getContext();

    SmallVector<mlir::Attribute> inputSyms(op.getInputs().size());
    SmallVector<mlir::Attribute> outputSyms(op.getOutputs().size());

    SmallVector<uint8_t> inputDimsBinaryVector, outputDimsBinaryVector;
    SmallVector<uint8_t> inputStridesBinaryVector, outputStridesBinaryVector;

    for (auto inputIt : llvm::enumerate(op.getInputs())) {
        auto inputIdx = inputIt.index();
        auto symVal = findSym(inputIt.value());
        inputSyms[inputIdx] = symVal;

        auto inVal = inputIt.value();
        auto inputNdType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(inVal.getType());
        VPUX_THROW_UNLESS(inputNdType, "Failed to cast to NDTypeInterface {0}", inVal);

        VPUASM::insertBinaryDimsIntoVector(inputDimsBinaryVector, inputNdType);
        VPUASM::insertBinaryStridesIntoVector(inputStridesBinaryVector, inputNdType);
    }

    bool skipBinaryOutput = false;
    for (auto outputIt : llvm::enumerate(op.getOutputs())) {
        auto outputIdx = outputIt.index();
        auto symVal = findSym(outputIt.value());
        outputSyms[outputIdx] = symVal;

        auto outVal = outputIt.value();
        auto outputNdType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(outVal.getType());
        VPUX_THROW_UNLESS(outputNdType, "Failed to cast to NDTypeInterface {0}", outVal);

        if (!skipBinaryOutput) {
            if (op.getIsOutputBroadcasted()) {
                skipBinaryOutput = true;
            }
            VPUASM::insertBinaryDimsIntoVector(outputDimsBinaryVector, outputNdType);
            VPUASM::insertBinaryStridesIntoVector(outputStridesBinaryVector, outputNdType);
        }
    }

    auto inputsAttr = mlir::ArrayAttr::get(context, inputSyms);
    auto outputsAttr = mlir::ArrayAttr::get(context, outputSyms);

    auto inputShapes = op.getDynamicInputShapes();
    auto outputShapes = op.getDynamicOutputShapes();

    auto [inputsShapeAttr, outputsShapeAttr] = processDynamicShapes(context, inputShapes, outputShapes);

    auto kernelParams = parseIntArrayAttr<uint8_t>(op.getKernelParams());

    auto newOp = rewriter.create<VPUASM::KernelParamsOp>(
            op.getLoc(), symName, inputsAttr, outputsAttr, inputsShapeAttr, outputsShapeAttr, op.getKernelTypeAttr(),
            std::move(kernelParams), std::move(inputDimsBinaryVector), std::move(inputStridesBinaryVector),
            std::move(outputDimsBinaryVector), std::move(outputStridesBinaryVector), op.getIsOutputBroadcastedAttr(),
            op.getIsJitCompiledAttr());
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

llvm::SmallVector<mlir::FlatSymbolRefAttr> KernelParamsRewriter::getSymbolicNames(VPUMI40XX::KernelParamsOp op,
                                                                                  size_t) {
    return createSymbolicName(op);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
