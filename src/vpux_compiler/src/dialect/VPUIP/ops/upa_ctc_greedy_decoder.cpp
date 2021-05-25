//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/utils/subspaces.hpp"

#include <mlir/IR/BuiltinTypes.h>

using namespace vpux;

mlir::LogicalResult vpux::VPUIP::verifyOp(CTCGreedyDecoderUPAOp op) {
    const auto inShape = getShape(op.input());

    if (inShape.size() != 3) {
        return errorAt(op, "Input shape should have 3 dimensions");
    }

    if (inShape[Dim(1)] != 1) {
        return errorAt(op, "Input tensor [T N C] = [{0} {1} {2}] has unsupported dimension size N != 1",
                       inShape[Dim(0)], inShape[Dim(1)], inShape[Dim(2)]);
    }

    return mlir::success();
}

mlir::LogicalResult vpux::VPUIP::CTCGreedyDecoderUPAOp::isSupportedLayout(mlir::Operation* op,
                                                                          vpux::DataOrderInfo& info) {
    const auto ctcGreedyDecoderOp = mlir::dyn_cast<IERT::CTCGreedyDecoderOp>(op);
    VPUX_THROW_UNLESS(ctcGreedyDecoderOp != nullptr, "Operation {0} is not CTCGreedyDecoderOp", op->getName());

    if (info.hasInput(0)) {
        const auto order = info.getInput(0);
        if (order == DimsOrder::CHW) {
            return mlir::success();
        }
    }

    info.setInput(0, DimsOrder::CHW);
    return mlir::failure();
}

void vpux::VPUIP::CTCGreedyDecoderUPAOp::build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::Value input,
                                               mlir::Value sequenceLengths, mlir::Value output,
                                               mlir::UnitAttr mergeRepeated) {
    build(builder, state, input, sequenceLengths, output, mlir::ValueRange{}, mlir::ValueRange{}, mergeRepeated,
          nullptr, nullptr);
}

VPUIP::BlobWriter::SpecificTask vpux::VPUIP::CTCGreedyDecoderUPAOp::serialize(VPUIP::BlobWriter& writer) {
    MVCNN::CTCDecoderParamsBuilder builder(writer);
    builder.add_ctc_merge_repeated(mergeRepeated());
    const auto paramsOff = builder.Finish();

    return writer.createUPALayerTask(*this, {paramsOff.Union(), MVCNN::SoftwareLayerParams_CTCDecoderParams});
}
