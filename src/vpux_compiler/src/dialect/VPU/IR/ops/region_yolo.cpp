//
// Copyright (C) 2022-2025 Intel Corporation.
// SPDX-License-Identifier: Apache-2.0
//

#include "vpux/compiler/dialect/VPU/IR/ops/specialized.hpp"

#include "vpux/utils/core/checked_cast.hpp"

using namespace vpux;

mlir::LogicalResult vpux::VPU::RegionYoloOp::inferReturnTypes(mlir::MLIRContext* ctx,
                                                              std::optional<mlir::Location> optLoc,
                                                              mlir::ValueRange operands, mlir::DictionaryAttr attrs,
                                                              mlir::OpaqueProperties prop,
                                                              mlir::RegionRange /*regions*/,
                                                              mlir::SmallVectorImpl<mlir::Type>& inferredReturnTypes) {
    const auto loc = optLoc.value_or(mlir::UnknownLoc::get(ctx));

    VPU::RegionYoloOpAdaptor regionYolo(operands, attrs, prop);
    if (mlir::failed(regionYolo.verify(loc))) {
        return mlir::failure();
    }

    const auto inType = mlir::cast<vpux::NDTypeInterface>(regionYolo.getInput().getType());

    SmallVector<int64_t> outputShape;
    if (regionYolo.getDoSoftmax()) {
        for (int64_t i = 0; i < regionYolo.getAxis(); i++) {
            outputShape.push_back(inType.getShape().raw()[i]);
        }

        size_t flat_dim = 1;
        for (int64_t i = regionYolo.getAxis(); i < regionYolo.getEndAxis() + 1; i++) {
            flat_dim *= inType.getShape().raw()[i];
        }
        outputShape.push_back(flat_dim);

        for (size_t i = regionYolo.getEndAxis() + 1; i < inType.getShape().size(); i++) {
            outputShape.push_back(inType.getShape().raw()[i]);
        }
    } else {
        outputShape.push_back(inType.getShape().raw()[0]);
        outputShape.push_back((regionYolo.getClasses() + regionYolo.getCoords() + 1) *
                              checked_cast<int64_t>(regionYolo.getMask().size()));
        outputShape.push_back(inType.getShape().raw()[2]);
        outputShape.push_back(inType.getShape().raw()[3]);
    }

    const auto outType = inType.changeShape(ShapeRef(outputShape));
    inferredReturnTypes.push_back(outType);

    return mlir::success();
}
