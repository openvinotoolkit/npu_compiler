//
// Copyright (C) 2023-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI37XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/utils/core/error.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

namespace vpux {
namespace vpumi37xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelParamsRewriter::symbolize(VPUMI37XX::KernelParamsOp op, SymbolMapper&,
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

    for (auto outputIt : llvm::enumerate(op.getOutputs())) {
        auto outputIdx = outputIt.index();
        auto symVal = findSym(outputIt.value());
        outputSyms[outputIdx] = symVal;

        auto outVal = outputIt.value();
        auto outputNdType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(outVal.getType());
        VPUX_THROW_UNLESS(outputNdType, "Failed to cast to NDTypeInterface {0}", outVal);

        VPUASM::insertBinaryDimsIntoVector(outputDimsBinaryVector, outputNdType);
        VPUASM::insertBinaryStridesIntoVector(outputStridesBinaryVector, outputNdType);
    }

    auto inputsAttr = mlir::ArrayAttr::get(context, inputSyms);
    auto outputsAttr = mlir::ArrayAttr::get(context, outputSyms);

    auto inputShapes = op.getDynamicInputShapes();
    auto outputShapes = op.getDynamicOutputShapes();

    auto [inputsShapeAttr, outputsShapeAttr] = processDynamicShapes(context, inputShapes, outputShapes);

    auto params = op.getKernelParamsAttr();
    auto denseElemData = params.getValues<uint8_t>();
    auto dataVector = SmallVector<uint8_t>(denseElemData.begin(), denseElemData.end());

    auto newOp = rewriter.create<VPUASM::KernelParamsOp>(
            op.getLoc(), symName, inputsAttr, outputsAttr, inputsShapeAttr, outputsShapeAttr, op.getKernelTypeAttr(),
            std::move(dataVector), std::move(inputDimsBinaryVector), std::move(inputStridesBinaryVector),
            std::move(outputDimsBinaryVector), std::move(outputStridesBinaryVector), nullptr, nullptr);

    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi37xx2vpuasm
}  // namespace vpux
