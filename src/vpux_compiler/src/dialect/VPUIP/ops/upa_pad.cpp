//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"

#include "vpux/compiler/dialect/VPUIP/graph-schema/blob_reader.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/utils.hpp"
#include "vpux/compiler/utils/error.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(PadUPAOp op) {
    const auto inShape = getShape(op.input());

    if (inShape.size() != op.pads_begin().size()) {
        return errorAt(op, "pads_begin attr size is not compatible with input tensor."
                           "The length of the list must be equal to the number of dimensions in the input tensor");
    }

    if (inShape.size() != op.pads_end().size()) {
        return errorAt(op, "pads_end attr size is not compatible with input tensor."
                           "The length of the list must be equal to the number of dimensions in the input tensor");
    }

    const auto padBegin = parseIntArrayAttr<int64_t>(op.pads_begin());
    const auto padEnd = parseIntArrayAttr<int64_t>(op.pads_end());
    if (op.pads_begin().size() == 4 && op.pads_end().size() == 4 && padEnd[0] - padBegin[0] != 0) {
        return errorAt(op, "PadUPAOp: Cannot expand batch");
    }

    if (op.mode() == IE::PadMode::CONSTANT && !op.pad_value().hasValue()) {
        return errorAt(op, "pad_mode is CONSTANT but pad_value hasn't provided");
    }

    return mlir::success();
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::PadUPAOp::serialize(VPUIP::BlobWriter& writer) {
    const auto padsBegin = writer.createVector(parseIntArrayAttr<uint32_t>(pads_begin()));
    const auto padsEnd = writer.createVector(parseIntArrayAttr<uint32_t>(pads_end()));

    MVCNN::PadParamsBuilder builder(writer);
    const auto padMode = VPUIP::convertVPUXPadMode2MVCNN(mode());
    builder.add_pad_mode(padMode);
    if (padMode == MVCNN::PadMode::PadMode_Constant) {
        builder.add_padValue(static_cast<float>(pad_value()->convertToDouble()));
    }
    builder.add_pads_begin(padsBegin);
    builder.add_pads_end(padsEnd);

    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_PadParams});
}

mlir::Operation* vpux::VPUIP::BlobReader::parsePad(mlir::OpBuilder& builder, ArrayRef<mlir::Value> inputs,
                                                   ArrayRef<mlir::Value> outputs, const MVCNN::UPALayerTask* task) {
    VPUX_THROW_UNLESS(inputs.size() == 1, "UPAPad supports only 1 input, got {0}", inputs.size());
    VPUX_THROW_UNLESS(outputs.size() == 1, "UPAPad supports only 1 output, got {0}", outputs.size());
    const auto params = task->softLayerParams_as_PadParams();
    const SmallVector<uint32_t> padBegin{params->pads_begin()->cbegin(), params->pads_begin()->cend()};
    const SmallVector<uint32_t> padEnd{params->pads_end()->cbegin(), params->pads_end()->cend()};
    const auto padValue = getFPAttr(_ctx, params->padValue());
    IE::PadMode padMode;
    switch (params->pad_mode()) {
    case MVCNN::PadMode::PadMode_Edge:
        padMode = IE::PadMode::EDGE;
        break;
    case MVCNN::PadMode::PadMode_Reflect:
        padMode = IE::PadMode::REFLECT;
        break;
    case MVCNN::PadMode::PadMode_Constant:
        padMode = IE::PadMode::CONSTANT;
        break;
    case MVCNN::PadMode::PadMode_Symmetric:
        padMode = IE::PadMode::SYMMETRIC;
        break;
    default:
        VPUX_THROW("Unsupported PadMode {0}", params->pad_mode());
    }

    return builder.create<VPUIP::PadUPAOp>(mlir::UnknownLoc::get(_ctx), inputs[0], outputs[0],
                                           getIntArrayAttr(_ctx, padBegin), getIntArrayAttr(_ctx, padEnd), padValue,
                                           IE::PadModeAttr::get(_ctx, padMode));
}
