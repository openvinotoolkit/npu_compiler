//
// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/conversion/rewriters/VPUMI40XX2VPUASM/kernel_params_rewriter.hpp"
#include "vpux/compiler/dialect/VPUASM/ops.hpp"
#include "vpux/compiler/dialect/VPUASM/utils.hpp"
#include "vpux/compiler/dialect/VPUIP/utils/sw_utils.hpp"
#include "vpux/compiler/utils/attributes.hpp"

#include <mlir/IR/BuiltinAttributes.h>

namespace vpux {
namespace vpumi40xx2vpuasm {

mlir::FailureOr<SymbolizationResult> KernelParamsRewriter::symbolize(VPUMI40XX::KernelParamsOp op, SymbolMapper&,
                                                                     mlir::ConversionPatternRewriter& rewriter) const {
    auto symName = findSym(op).getRootReference();
    auto context = getContext();

    // Cache ranges to avoid repeated proxy object creation from op.getInputs()/op.getOutputs()
    const auto& inputs = op.getInputs();
    const auto& outputs = op.getOutputs();

    SmallVector<mlir::Attribute> inputSyms(inputs.size());
    SmallVector<mlir::Attribute> outputSyms(outputs.size());

    SmallVector<uint8_t> inputDimsBinaryVector, outputDimsBinaryVector;
    SmallVector<uint8_t> inputStridesBinaryVector, outputStridesBinaryVector;

    // Kernels in SW_KERNELS_SUPPORTING_STRIDE navigate memory using the stride values directly and
    // may derive dimension sizes from stride ratios. Compact normalization of unit-size strides
    // would corrupt those ratios, so such kernels must receive the original (possibly inherited)
    // strides unchanged. All other kernels require compact strides.
    const bool normalizeUnitDimStrides = !llvm::is_contained(VPUIP::SW_KERNELS_SUPPORTING_STRIDE, op.getKernelType());

    for (const auto& [inputIdx, inputVal] : llvm::enumerate(inputs)) {
        inputSyms[inputIdx] = findSym(inputVal);

        auto inputNdType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(inputVal.getType());
        VPUX_THROW_UNLESS(inputNdType, "Failed to cast to NDTypeInterface {0}", inputVal);

        VPUASM::insertBinaryDimsIntoVector(inputDimsBinaryVector, inputNdType);
        VPUASM::insertBinaryStridesIntoVector(inputStridesBinaryVector, inputNdType, normalizeUnitDimStrides);
    }

    // Cache this boolean to avoid repeated method calls in the loop
    const bool isOutputBroadcasted = op.getIsOutputBroadcasted();
    bool skipBinaryOutput = false;
    for (const auto& [outputIdx, outputVal] : llvm::enumerate(outputs)) {
        outputSyms[outputIdx] = findSym(outputVal);

        auto outputNdType = mlir::dyn_cast_or_null<vpux::NDTypeInterface>(outputVal.getType());
        VPUX_THROW_UNLESS(outputNdType, "Failed to cast to NDTypeInterface {0}", outputVal);

        if (!skipBinaryOutput) {
            if (isOutputBroadcasted) {
                skipBinaryOutput = true;
            }
            VPUASM::insertBinaryDimsIntoVector(outputDimsBinaryVector, outputNdType);
            VPUASM::insertBinaryStridesIntoVector(outputStridesBinaryVector, outputNdType, normalizeUnitDimStrides);
        }
    }

    auto inputsAttr = mlir::ArrayAttr::get(context, inputSyms);
    auto outputsAttr = mlir::ArrayAttr::get(context, outputSyms);

    // Cache dynamic shapes to avoid repeated temporary object creation
    auto inputShapes = op.getDynamicInputShapes();
    auto outputShapes = op.getDynamicOutputShapes();

    auto [inputsShapeAttr, outputsShapeAttr] = processDynamicShapes(context, inputShapes, outputShapes);

    auto kernelParams = parseIntArrayAttr<uint8_t>(op.getKernelParams());
    // Simplify skipDescIds logic: use direct access when has_value, otherwise nullptr
    auto skipDescIds = op.getSkipDescIds().value_or(nullptr);

    auto newOp = rewriter.create<VPUASM::KernelParamsOp>(
            op.getLoc(), symName, inputsAttr, outputsAttr, inputsShapeAttr, outputsShapeAttr, op.getKernelTypeAttr(),
            std::move(kernelParams), std::move(inputDimsBinaryVector), std::move(inputStridesBinaryVector),
            std::move(outputDimsBinaryVector), std::move(outputStridesBinaryVector), isOutputBroadcasted,
            op.getIsJitCompiled(), op.getUsesDma(), skipDescIds);
    rewriter.eraseOp(op);

    return SymbolizationResult(newOp);
}

}  // namespace vpumi40xx2vpuasm
}  // namespace vpux
