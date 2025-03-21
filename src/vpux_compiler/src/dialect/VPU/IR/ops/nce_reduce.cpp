// Copyright (C) 2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "vpux/compiler/dialect/VPU/IR/ops.hpp"

#include "vpux/compiler/dialect/VPU/utils/const_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/nce_reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/reduce_utils.hpp"
#include "vpux/compiler/dialect/VPU/utils/type_infer.hpp"
#include "vpux/compiler/utils/attributes.hpp"

using namespace vpux;

//
// InferTypeOpInterface
//

mlir::LogicalResult vpux::VPU::NCEReduceOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                             std::optional<mlir::Location> optLoc,
                                                             mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                             mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                             mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::NCEReduceOpAdaptor reduce(operands, attrs, prop);
    if (mlir::failed(reduce.verify(loc))) {
        return mlir::failure();
    }

    const auto input = reduce.getInput();
    auto axes = parseIntArrayAttr<int64_t>(reduce.getAxesAttr());

    return VPU::inferReduceReturnTypes(loc, input, /*keep_dims*/ true, /*axes*/ axes, inferredReturnTypes);
}

//
// isSupported
//

bool vpux::VPU::NCEReduceOp::isSupported(mlir::Operation* op, LogCb logCb, bool checkLayout,
                                         bool checkChannelAlignment) {
    if (!isReduceOpSupportedOnNCE(op) || !vpux::VPU::isNCEReduceSupported(op, logCb)) {
        return false;
    }

    auto inputType = op->getOperand(0).getType().cast<vpux::NDTypeInterface>();
    auto outputType = op->getResult(0).getType().cast<vpux::NDTypeInterface>();

    if (inputType.getRank() != 4 || outputType.getRank() != 4) {
        logCb(formatv("Only 4D tensors are supported"));
        return false;
    }

    if (checkChannelAlignment) {
        if (auto iface = mlir::dyn_cast<IE::AlignedChannelsOpInterface>(op)) {
            if (!NCEInvariant::isInputActTypeSupported(inputType, iface.getInputChannelAlignment(), false) ||
                !NCEInvariant::isOutputActTypeSupported(outputType, iface.getOutputChannelAlignment())) {
                logCb(formatv("Misaligned tensor shape"));
                return false;
            }
        }
    }

    if (checkLayout) {
        if (!NCEInvariant::checkLayouts({inputType}, {outputType}, getArch(op), 1, logCb)) {
            return false;
        }
    }
    return true;
}
