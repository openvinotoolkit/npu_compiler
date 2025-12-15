//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/IE/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/IR/ops/recurrent.hpp"
#include "vpux/compiler/dialect/VPU/transforms/factories/max_lstm_hidden_size_constant.hpp"
#include "vpux/compiler/dialect/config/IR/utils.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::LSTMCellOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                            std::optional<mlir::Location> optLoc,
                                                            mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                            mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                            mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::LSTMCellOpAdaptor lstm(operands, attrs, prop);
    if (mlir::failed(lstm.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(lstm.getInitialHiddenState().getType());

    inferredReturnTypes.push_back(inType);  // outputHiddenState
    inferredReturnTypes.push_back(inType);  // outputCellState

    return mlir::success();
}

namespace {

bool isSupported(config::ArchKind arch, ShapeRef inputDataShape, ShapeRef initialHiddenStateShape) {
    auto maxHiddenSize = VPU::getMaxLstmCellHiddenSizeConstant(arch);
    // shave implementation allow reduced size. Bigger size can be map on DPU.
    // Cost model can be interrogate.
    constexpr int64_t maxInputSize(256);
    constexpr int64_t maxBatchSize(1);
    // shave asm implement just 16 element alignment input and hidden size. Except that speed is low.
    constexpr int64_t alignmentRequired(16);
    if ((inputDataShape.back() > maxInputSize) || (initialHiddenStateShape.back() > maxHiddenSize) ||
        (inputDataShape[Dim(inputDataShape.size() - 2)] > maxBatchSize)) {
        return false;
    }
    if ((inputDataShape.back() % alignmentRequired != 0) || (initialHiddenStateShape.back() % alignmentRequired != 0)) {
        return false;
    }
    return true;
}

}  // namespace

//
// isSupported
//

bool vpux::VPU::LSTMCellOp::isSupported(vpux::IE::LSTMCellOp op) {
    return ::isSupported(config::getArch(op), getShape(op.getInputData()), getShape(op.getInitialHiddenState()));
}
