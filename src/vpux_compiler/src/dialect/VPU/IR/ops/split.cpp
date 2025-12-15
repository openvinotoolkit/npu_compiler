//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/data_movement.hpp"

#include "vpux/compiler/dialect/const/ops.hpp"
#include "vpux/compiler/utils/error.hpp"

using namespace vpux;

namespace {

Dim normalizeAxis(VPU::SplitOpAdaptor split) {
    VPUX_THROW_UNLESS(split.getAxisValue().has_value(), "Got non constant axis");

    const auto inType = mlir::cast<vpux::NDTypeInterface>(split.getInput().getType());
    const auto inRank = inType.getRank();

    auto axisInd = split.getAxisValue().value();

    // Negative value means counting dimension from the end
    if (axisInd < 0) {
        axisInd += inRank;
    }

    VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                      inRank);

    return Dim(axisInd);
}

mlir::FailureOr<Dim> extractAxis(mlir::Location loc, VPU::SplitOpAdaptor split) {
    if (split.getAxis() != nullptr) {
        auto axisValue = split.getAxis();
        while (auto parentOp = axisValue.getDefiningOp<VPU::CopyOp>()) {
            axisValue = parentOp->getOperand(0);
        }
        auto axisConst = axisValue.getDefiningOp<Const::DeclareOp>();
        if (axisConst == nullptr) {
            return errorAt(loc, "Only constant input is supported for axis");
        }

        if (const auto& attr = axisConst.getContentAttr(); !attr.isSplat()) {
            return errorAt(loc, "Axis value must be a scalar");
        }

        const auto inType = mlir::cast<vpux::NDTypeInterface>(split.getInput().getType());
        const auto inRank = inType.getRank();

        const auto axisContent = axisConst.getContent();
        auto axisInd = axisContent.getSplatValue<int64_t>();

        // Negative value means counting dimension from the end
        if (axisInd < 0) {
            axisInd += inRank;
        }

        VPUX_THROW_UNLESS(axisInd >= 0 && axisInd < inRank, "Got wrong Split axis '{0}', out of range '{1}'", axisInd,
                          inRank);

        return Dim(axisInd);
    } else if (split.getAxisValue().has_value()) {
        return normalizeAxis(split);
    } else {
        return errorAt(loc, "Axis was not provided");
    }
}

}  // namespace

mlir::LogicalResult vpux::VPU::SplitOp::inferReturnTypes(mlir::MLIRContext* ctx, std::optional<mlir::Location> optLoc,
                                                         mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                         mlir::OpaqueProperties prop, mlir::RegionRange /*regions*/,
                                                         mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::SplitOpAdaptor split(operands, attrs, prop);
    if (mlir::failed(split.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(split.getInput().getType());

    const auto axis = extractAxis(loc, split);
    if (mlir::failed(axis)) {
        return mlir::failure();
    }

    const auto num_splits = split.getNumSplits();
    if (num_splits <= 0) {
        return errorAt(loc, "Number of splits should be a natural number");
    }

    auto outShape = mlir::cast<vpux::NDTypeInterface>(inType).getShape().toValues();
    if ((outShape[*axis] < num_splits) || (outShape[*axis] % num_splits != 0)) {
        return errorAt(loc, "Unsupported num_splits parameter");
    }
    outShape[*axis] /= num_splits;

    for (int i = 0; i < num_splits; ++i) {
        const auto outType = inType.changeShape(outShape);
        inferredReturnTypes.push_back(outType);
    }

    return mlir::success();
}

//
// verify
//

mlir::LogicalResult vpux::VPU::SplitOp::verify() {
    const auto inType = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(getInput().getType());
    if (inType != nullptr && inType.containsDistributedTypes()) {
        return errorAt(*this, "Split op cannot have Distributed input type", inType);
    }

    for (const auto& output : getOutputs()) {
        auto outType = mlir::dyn_cast<vpux::VPU::DistributedTypeInterface>(output.getType());
        if (outType != nullptr && outType.containsDistributedTypes()) {
            return errorAt(*this, "Split op cannot have Distributed output type", outType);
        }
    }

    if (getNumSplits() <= 0) {
        return errorAt(*this, "Number of splits should be a positive integer, while it is {0}", getNumSplits());
    }

    return mlir::success();
}
